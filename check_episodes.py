import logging
from reinforcestrategycreator.db_utils import get_db_session
from reinforcestrategycreator.db_models import Episode

logging.basicConfig(level=logging.INFO)

with get_db_session() as db:
    incomplete_episodes = db.query(Episode).filter(Episode.end_time.is_(None)).count()
    print(f'Episodes with no end_time: {incomplete_episodes}')
    
    total_episodes = db.query(Episode).count()
    print(f'Total episodes: {total_episodes}')
    
    if total_episodes > 0:
        completion_rate = (total_episodes - incomplete_episodes) / total_episodes * 100
        print(f'Episode completion rate: {completion_rate:.2f}%')