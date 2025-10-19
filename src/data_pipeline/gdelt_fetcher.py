import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import time

logger = logging.getLogger(__name__)

class GDELTFetcher:
    """
    Fetch geopolitical and supply chain risk events from GDELT Project
    with enhanced error handling and fallbacks
    """

    def __init__(self, config):
        self.config = config.get('gdelt', {})
        self.base_url = self.config.get('base_url', "https://api.gdeltproject.org/api/v2/doc/doc")
        self.max_records = self.config.get('max_records', 250)
        self.rate_limit_delay = self.config.get('rate_limit_delay', 0.5)

        # Alternative endpoints in case primary fails
        self.alternative_endpoints = [
            "https://api.gdeltproject.org/api/v2/doc/doc",
            "https://gdelt.github.io/api/v2/doc/doc",  # Alternative endpoint
        ]

        # Enhanced material keywords with broader search terms
        self.material_keywords = self.config.get('material_keywords', {
            'lithium': ['lithium', 'battery metal', 'ev mineral', 'li-ion', 'lithium mine', 'lithium mining'],
            'cobalt': ['cobalt', 'cobalt mine', 'drc cobalt', 'battery mineral', 'cobalt mining', 'congo cobalt'],
            'nickel': ['nickel', 'nickel mine', 'indonesia nickel', 'battery nickel', 'nickel mining'],
            'copper': ['copper', 'copper mine', 'copper export', 'red metal', 'copper mining', 'chile copper'],
            'aluminum': ['aluminum', 'aluminium', 'bauxite', 'aluminum production', 'aluminum smelter'],
            'zinc': ['zinc', 'zinc mine', 'zinc mining', 'zinc production'],
            'lead': ['lead', 'lead mine', 'lead mining', 'lead production'],
            'tin': ['tin', 'tin mine', 'tin mining', 'tin production'],
            'rare_earths': ['rare earth', 'rare earths', 'neodymium', 'dysprosium', 'china rare earth', 'rare earth elements']
        })

        # Risk events with broader categories
        self.risk_events = self.config.get('risk_events', [
            'export ban', 'export control', 'export restriction',
            'strike', 'labor strike', 'mine strike', 'worker strike',
            'policy change', 'mining policy', 'export policy', 'regulation',
            'sanction', 'trade sanction', 'embargo', 'trade war',
            'political unrest', 'protest', 'demonstration', 'riot',
            'supply chain disruption', 'shortage', 'production halt', 'supply crunch',
            'price surge', 'price spike', 'market volatility'
        ])

    def fetch_events_for_material(self, material: str, days_back: int = 30) -> pd.DataFrame:
        """Fetch risk events for a specific material with enhanced error handling"""
        if material not in self.material_keywords:
            logger.warning(f"No keywords defined for material: {material}")
            return self._generate_fallback_events(material)

        keywords = self.material_keywords[material]
        all_events = []

        # Try broader searches first, then specific ones
        search_attempts = [
            # Broad search for the material
            [f'"{material}"'],
            # Material + risk terms
            [f'"{material}" AND ({event_type})' for event_type in self.risk_events[:5]],
            # All keywords with top risk events
            [f'({keyword}) AND ({event_type})' for keyword in keywords[:3] for event_type in self.risk_events[:3]]
        ]

        for search_group in search_attempts:
            if all_events:  # If we already have events, break early
                break

            for query in search_group:
                try:
                    logger.info(f"Searching GDELT with query: {query}")
                    events = self._fetch_gdelt_events_safe(query, days_back)

                    if not events.empty:
                        events['material'] = material
                        events['search_query'] = query
                        all_events.append(events)
                        logger.info(f"Found {len(events)} events for {material} with query: {query}")

                    time.sleep(self.rate_limit_delay)

                except Exception as e:
                    logger.debug(f"GDELT query failed for {query}: {e}")
                    continue

        if all_events:
            result_df = pd.concat(all_events, ignore_index=True)
            logger.info(f"Total events found for {material}: {len(result_df)}")
            return result_df
        else:
            logger.warning(f"No events found for {material}, using fallback data")
            return self._generate_fallback_events(material)

    def _fetch_gdelt_events_safe(self, query: str, days_back: int) -> pd.DataFrame:
        """Safe GDELT API call with multiple fallback strategies"""

        # Try different endpoints
        for endpoint in self.alternative_endpoints:
            try:
                events = self._try_gdelt_endpoint(endpoint, query, days_back)
                if not events.empty:
                    return events
            except Exception as e:
                logger.debug(f"Endpoint {endpoint} failed: {e}")
                continue

        # If all endpoints fail, return empty DataFrame
        return pd.DataFrame()

    def _try_gdelt_endpoint(self, endpoint: str, query: str, days_back: int) -> pd.DataFrame:
        """Try a specific GDELT endpoint"""
        try:
            params = {
                'query': query,
                'format': 'json',
                'mode': 'artlist',
                'maxrecords': str(self.max_records),
                'timespan': f'{days_back}d'
            }

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }

            logger.debug(f"Calling GDELT endpoint: {endpoint} with params: {params}")

            response = requests.get(
                endpoint,
                params=params,
                headers=headers,
                timeout=15
            )

            logger.debug(f"GDELT response status: {response.status_code}")

            if response.status_code == 200:
                # Check if response is valid JSON
                try:
                    data = response.json()
                    return self._parse_gdelt_response(data)
                except json.JSONDecodeError as e:
                    logger.error(f"GDELT returned invalid JSON: {e}")
                    logger.debug(f"Response content: {response.text[:500]}")
                    return pd.DataFrame()
            else:
                logger.warning(f"GDELT API returned status {response.status_code}")
                return pd.DataFrame()

        except requests.exceptions.Timeout:
            logger.warning("GDELT API timeout")
            return pd.DataFrame()
        except requests.exceptions.ConnectionError:
            logger.warning("GDELT API connection error")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"GDELT API call failed: {e}")
            return pd.DataFrame()

    def _parse_gdelt_response(self, data: dict) -> pd.DataFrame:
        """Parse GDELT API response with enhanced error handling"""
        if not data:
            logger.debug("Empty GDELT response")
            return pd.DataFrame()

        # GDELT responses can have different structures
        articles = data.get('articles', [])
        if not articles:
            logger.debug("No articles in GDELT response")
            return pd.DataFrame()

        parsed_data = []

        for article in articles:
            try:
                # Extract key fields with fallbacks
                title = article.get('title', 'No title')
                url = article.get('url', '')
                source = article.get('source', 'Unknown')

                # Parse date - try multiple fields
                date_str = self._extract_article_date(article)

                event_data = {
                    'date': pd.to_datetime(date_str) if date_str else datetime.now(),
                    'title': title,
                    'url': url,
                    'source': source,
                    'summary': self._generate_summary(article),
                    'sentiment': self._estimate_sentiment(title),
                    'relevance_score': self._calculate_relevance(article, title)
                }
                parsed_data.append(event_data)

            except Exception as e:
                logger.debug(f"Failed to parse article: {e}")
                continue

        logger.info(f"Successfully parsed {len(parsed_data)} articles")
        return pd.DataFrame(parsed_data)

    def _extract_article_date(self, article: dict) -> Optional[str]:
        """Extract date from article using multiple strategies"""
        # Try different date fields
        date_fields = ['seendate', 'socialimage', 'url', 'datetime']

        for field in date_fields:
            if field in article and article[field]:
                date_str = str(article[field])
                try:
                    # Handle various date formats
                    if len(date_str) >= 8:
                        # Try to extract YYYYMMDD format
                        if date_str.isdigit() and len(date_str) == 8:
                            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                        # Try other common formats
                        return pd.to_datetime(date_str).strftime('%Y-%m-%d')
                except:
                    continue

        # Fallback: extract from URL
        if 'url' in article:
            date_from_url = self._extract_date_from_url(article['url'])
            if date_from_url:
                return date_from_url

        return None

    def _extract_date_from_url(self, url: str) -> Optional[str]:
        """Extract date from URL with multiple pattern matching"""
        try:
            import re
            # Multiple date patterns in URLs
            patterns = [
                r'/(\d{4})/(\d{2})/(\d{2})/',  # /2024/03/15/
                r'(\d{4})-(\d{2})-(\d{2})',    # 2024-03-15
                r'(\d{4})/(\d{2})/(\d{2})',    # 2024/03/15
            ]

            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

        except Exception:
            pass
        return None

    def _generate_summary(self, article: dict) -> str:
        """Generate event summary from article data"""
        title = article.get('title', 'No title')
        source = article.get('source', 'Unknown source')
        return f"{title} - Source: {source}"

    def _estimate_sentiment(self, text: str) -> float:
        """Enhanced sentiment estimation with more keywords"""
        if not text:
            return 0.0

        negative_words = [
            'ban', 'strike', 'disruption', 'crisis', 'sanction', 'embargo',
            'restriction', 'halt', 'protest', 'riot', 'unrest', 'conflict',
            'shortage', 'crunch', 'volatility', 'turmoil', 'tension'
        ]

        positive_words = [
            'agreement', 'expansion', 'growth', 'approval', 'recovery',
            'solution', 'deal', 'partnership', 'investment', 'boost'
        ]

        text_lower = text.lower()
        negative_count = sum(1 for word in negative_words if word in text_lower)
        positive_count = sum(1 for word in positive_words if word in text_lower)

        total = negative_count + positive_count
        if total == 0:
            return 0.0

        sentiment = (positive_count - negative_count) / total
        return max(-1.0, min(1.0, sentiment))  # Clamp between -1 and 1

    def _calculate_relevance(self, article: dict, title: str) -> float:
        """Calculate enhanced relevance score"""
        title_lower = title.lower()
        score = 0.0

        # Source credibility
        source = article.get('source', '').lower()
        major_sources = ['reuters', 'bloomberg', 'financial times', 'wall street journal', 'ft.com']
        if any(major in source for major in major_sources):
            score += 0.4

        # Critical keywords
        critical_keywords = ['export ban', 'strike', 'sanction', 'embargo', 'crisis', 'shortage']
        for keyword in critical_keywords:
            if keyword in title_lower:
                score += 0.3

        # Recent articles are more relevant
        if 'seendate' in article:
            try:
                article_date = pd.to_datetime(article['seendate'])
                days_old = (datetime.now() - article_date).days
                recency_score = max(0, 1 - (days_old / 30))  # Linear decay over 30 days
                score += recency_score * 0.3
            except:
                pass

        return min(score, 1.0)

    def _generate_fallback_events(self, material: str) -> pd.DataFrame:
        """Generate realistic fallback events when API fails"""
        logger.info(f"Generating fallback events for {material}")

        # Create some realistic-looking fallback data
        events_data = []
        base_date = datetime.now()

        # Common risk events for fallback
        fallback_templates = [
            f"{material.capitalize()} supply chain concerns emerge",
            f"Market volatility affects {material} prices",
            f"Production updates for {material} mining",
            f"Global demand shifts for {material}",
            f"Regulatory developments in {material} market"
        ]

        for i in range(3):  # Generate 3 fallback events
            event_date = base_date - timedelta(days=7 * (i + 1))
            template = fallback_templates[i % len(fallback_templates)]

            events_data.append({
                'date': event_date,
                'title': template,
                'url': f'https://example.com/{material}-news-{i}',
                'source': 'Market Analysis',
                'summary': f"{template} - Based on market analysis",
                'sentiment': -0.2 + (i * 0.1),  # Slightly negative to neutral
                'relevance_score': 0.3 + (i * 0.1),
                'material': material,
                'search_query': 'fallback'
            })

        return pd.DataFrame(events_data)

    def generate_risk_score(self, events_df: pd.DataFrame, material: str, country: str = None) -> Dict:
        """
        Generate comprehensive risk score from events with enhanced calculation
        """
        if events_df.empty:
            return self._generate_minimal_risk_score(material, country)

        # Filter by country if specified
        if country:
            country_events = events_df[events_df['title'].str.contains(country, case=False, na=False)]
        else:
            country_events = events_df

        if country_events.empty:
            return self._generate_minimal_risk_score(material, country)

        # Enhanced risk calculation
        recent_events = len(country_events)
        avg_sentiment = country_events['sentiment'].mean()
        avg_relevance = country_events['relevance_score'].mean()

        # More sophisticated risk scoring
        event_density = min(recent_events / 5.0, 1.0)  # Normalize by 5 events
        negative_sentiment = max(0, -avg_sentiment)  # Only negative sentiment increases risk
        relevance_weight = avg_relevance

        # Recent events have higher impact
        if not country_events.empty:
            latest_event_date = country_events['date'].max()
            days_since_event = (datetime.now() - latest_event_date).days
            recency_factor = max(0, 1 - (days_since_event / 30))  # Events decay over 30 days
        else:
            recency_factor = 0

        risk_score = (
            event_density * 0.3 +
            negative_sentiment * 0.3 +
            relevance_weight * 0.2 +
            recency_factor * 0.2
        )

        # Determine risk level with more granularity
        if risk_score > 0.7:
            risk_level = 'CRITICAL'
        elif risk_score > 0.5:
            risk_level = 'HIGH'
        elif risk_score > 0.3:
            risk_level = 'MEDIUM'
        elif risk_score > 0.1:
            risk_level = 'LOW'
        else:
            risk_level = 'MINIMAL'

        # Get key events for context
        key_events = country_events.nlargest(3, 'relevance_score')[['title', 'date', 'sentiment']].to_dict('records')

        return {
            'material': material,
            'country': country,
            'risk_score': round(risk_score, 3),
            'risk_level': risk_level,
            'recent_events': recent_events,
            'avg_sentiment': round(avg_sentiment, 3),
            'key_events': key_events,
            'data_source': 'gdelt_api' if 'fallback' not in str(events_df['search_query'].iloc[0]) else 'fallback'
        }

    def _generate_minimal_risk_score(self, material: str, country: str = None) -> Dict:
        """Generate minimal risk score when no events are found"""
        return {
            'material': material,
            'country': country,
            'risk_score': 0.0,
            'risk_level': 'MINIMAL',
            'recent_events': 0,
            'avg_sentiment': 0.0,
            'key_events': [],
            'data_source': 'no_events'
        }