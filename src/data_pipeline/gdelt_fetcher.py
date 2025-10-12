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
    """

    def __init__(self, config):
        self.config = config.get('gdelt', {})
        self.base_url = self.config.get('base_url', "https://api.gdeltproject.org/api/v2/doc/doc")
        self.max_records = self.config.get('max_records', 250)
        self.rate_limit_delay = self.config.get('rate_limit_delay', 0.5)

        # Use config-based keywords or fallback to defaults
        self.material_keywords = self.config.get('material_keywords', {
            'lithium': ['lithium', 'battery metal', 'ev mineral', 'li-ion'],
            'cobalt': ['cobalt', 'cobalt mine', 'drc cobalt', 'battery mineral'],
            'nickel': ['nickel', 'nickel mine', 'indonesia nickel', 'battery nickel'],
            'copper': ['copper', 'copper mine', 'copper export', 'red metal'],
            'rare_earths': ['rare earth', 'rare earths', 'neodymium', 'dysprosium', 'china rare earth']
        })

        # Use config-based risk events or fallback to defaults
        self.risk_events = self.config.get('risk_events', [
            'export ban', 'export control', 'export restriction',
            'strike', 'labor strike', 'mine strike',
            'policy change', 'mining policy', 'export policy',
            'sanction', 'trade sanction', 'embargo',
            'political unrest', 'protest', 'demonstration',
            'supply chain disruption', 'shortage', 'production halt'
        ])

    def fetch_events_for_material(self, material: str, days_back: int = 30) -> pd.DataFrame:
        """Fetch risk events for a specific material using config settings"""
        if material not in self.material_keywords:
            logger.warning(f"No keywords defined for material: {material}")
            return pd.DataFrame()

        keywords = self.material_keywords[material]
        all_events = []

        for keyword in keywords:
            for event_type in self.risk_events:
                try:
                    query = f"({keyword}) AND ({event_type})"
                    events = self._fetch_gdelt_events(query, days_back)
                    if not events.empty:
                        events['material'] = material
                        events['search_keyword'] = keyword
                        events['event_type'] = event_type
                        all_events.append(events)

                    # Respect rate limit from config
                    time.sleep(self.rate_limit_delay)

                except Exception as e:
                    logger.debug(f"GDELT query failed for {keyword}-{event_type}: {e}")
                    continue

        if all_events:
            return pd.concat(all_events, ignore_index=True)
        else:
            return pd.DataFrame()


    def _fetch_gdelt_events(self, query: str, days_back: int) -> pd.DataFrame:
        """
        Execute GDELT API query
        """
        try:
            params = {
                'query': query,
                'format': 'json',
                'mode': 'artlist',
                'maxrecords': 250,
                'timespan': f'{days_back}d'
            }

            response = requests.get(self.base_url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                return self._parse_gdelt_response(data)
            else:
                logger.warning(f"GDELT API returned status {response.status_code}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"GDELT API call failed: {e}")
            return pd.DataFrame()

    def _parse_gdelt_response(self, data: dict) -> pd.DataFrame:
        """
        Parse GDELT API response into structured DataFrame
        """
        if not data or 'articles' not in data:
            return pd.DataFrame()

        articles = data['articles']
        parsed_data = []

        for article in articles:
            try:
                # Extract event date (use URL date if available, otherwise current date)
                url = article.get('url', '')
                date_str = self._extract_date_from_url(url)

                event_data = {
                    'date': pd.to_datetime(date_str) if date_str else datetime.now(),
                    'title': article.get('title', ''),
                    'url': url,
                    'source': article.get('source', ''),
                    'summary': self._generate_summary(article),
                    'sentiment': self._estimate_sentiment(article.get('title', '') + ' ' + article.get('seendate', '')),
                    'relevance_score': self._calculate_relevance(article)
                }
                parsed_data.append(event_data)

            except Exception as e:
                logger.debug(f"Failed to parse article: {e}")
                continue

        return pd.DataFrame(parsed_data)

    def _extract_date_from_url(self, url: str) -> Optional[str]:
        """Extract date from GDELT URL format"""
        try:
            # GDELT URLs often contain dates in format /2024/03/15/
            import re
            date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', url)
            if date_match:
                return f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
        except:
            pass
        return None

    def _generate_summary(self, article: dict) -> str:
        """Generate event summary from article data"""
        title = article.get('title', '')
        return f"{title} - Source: {article.get('source', 'Unknown')}"

    def _estimate_sentiment(self, text: str) -> float:
        """Simple sentiment estimation (-1 to 1)"""
        negative_words = ['ban', 'strike', 'disruption', 'crisis', 'sanction', 'embargo', 'restriction', 'halt']
        positive_words = ['agreement', 'expansion', 'growth', 'approval', 'recovery', 'solution']

        text_lower = text.lower()
        negative_count = sum(1 for word in negative_words if word in text_lower)
        positive_count = sum(1 for word in positive_words if word in text_lower)

        total = negative_count + positive_count
        if total == 0:
            return 0.0

        return (positive_count - negative_count) / total

    def _calculate_relevance(self, article: dict) -> float:
        """Calculate relevance score for the event"""
        title = article.get('title', '').lower()
        score = 0.0

        # Higher score for major news sources
        source = article.get('source', '').lower()
        major_sources = ['reuters', 'bloomberg', 'financial times', 'wall street journal']
        if any(major in source for major in major_sources):
            score += 0.3

        # Score based on keywords in title
        critical_keywords = ['export ban', 'strike', 'sanction', 'embargo', 'crisis']
        for keyword in critical_keywords:
            if keyword in title:
                score += 0.2

        return min(score, 1.0)

    def generate_risk_score(self, events_df: pd.DataFrame, material: str, country: str = None) -> Dict:
        """
        Generate comprehensive risk score from events
        """
        if events_df.empty:
            return {
                'material': material,
                'country': country,
                'risk_score': 0.0,
                'risk_level': 'LOW',
                'recent_events': 0,
                'avg_sentiment': 0.0,
                'key_events': []
            }

        # Filter by country if specified
        if country:
            country_events = events_df[events_df['title'].str.contains(country, case=False, na=False)]
        else:
            country_events = events_df

        if country_events.empty:
            return {
                'material': material,
                'country': country,
                'risk_score': 0.0,
                'risk_level': 'LOW',
                'recent_events': 0,
                'avg_sentiment': 0.0,
                'key_events': []
            }

        # Calculate metrics
        recent_events = len(country_events)
        avg_sentiment = country_events['sentiment'].mean()
        avg_relevance = country_events['relevance_score'].mean()

        # Risk score calculation (0-1 scale)
        event_density = min(recent_events / 10.0, 1.0)  # Cap at 10 events
        negative_sentiment = max(0, -avg_sentiment)  # Only negative sentiment increases risk
        risk_score = (event_density * 0.4) + (negative_sentiment * 0.4) + (avg_relevance * 0.2)

        # Determine risk level
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
            'risk_score': risk_score,
            'risk_level': risk_level,
            'recent_events': recent_events,
            'avg_sentiment': avg_sentiment,
            'key_events': key_events
        }