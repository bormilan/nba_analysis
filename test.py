import pandas as pd
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams

id = teams.find_teams_by_full_name('Chicago Bulls')[0]['id']

log = pd.DataFrame(teamgamelog.TeamGameLog(team_id=[id]).get_normalized_dict()["TeamGameLog"])
log.to_csv('chicago_stat.csv')