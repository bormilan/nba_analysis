from os import name
import pandas as pd
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_team_log(team):
    id = teams.find_teams_by_full_name(team)[0]['id']
    log = pd.DataFrame(teamgamelog.TeamGameLog(team_id=[id]).get_normalized_dict()["TeamGameLog"])

    return log

def prepare_data(log):
    log['WL'] = log['WL'].map(lambda x: 1 if x == 'W' else 0)
    new_log = log[::-1]
    new_log.reset_index(drop=True)
    new_log.index = np.arange(new_log['PTS'].count())

    return new_log

def make_plot_from_season_pts(log):
    pts = log['PTS']

    fig = px.line(pts)

    fig.show()

def make_regression(log):   
    #labels 
    y = log['PTS']
    #training data
    X = np.arange(y.count())[:,np.newaxis]
    #model
    model = LinearRegression().fit(X,y)
    
    #new predictors
    xfit = np.arange(y.count()+5)
    Xfit = xfit[:, np.newaxis]
    #predictand values
    yfit = model.predict(Xfit)

    #plot
        #train values
    plt.scatter(X, y)
        #predictor value
    plt.scatter(y.count()+5, 0, c='red', marker='x')
        #predictand value
    plt.scatter(y.count()+5, model.predict(np.array(5.8).reshape(-1,1)), c='red', marker='x')
        #new values
    plt.scatter(xfit, yfit)
        #plot a line
    plt.plot(xfit, yfit)

def make_histogram(log,by,color):

    fig = px.histogram(log, by, color=color)
    fig.show()

def make_boxplot_from_pts(log):

    fig = px.box(log, 'PTS', title="Deviation of the teams's scores")
    fig.show()

def clean_matchup(log):
    return log['MATCHUP'].map(lambda x: x.split(' ')[2])

def scores_by_opponent(log):
    by_opponent = pd.DataFrame()
    by_opponent['count'] = log.groupby('MATCHUP')['PTS'].count()
    by_opponent['min'] = log.groupby('MATCHUP')['PTS'].min()
    by_opponent['max'] = log.groupby('MATCHUP')['PTS'].max()
    by_opponent['mean'] = log.groupby('MATCHUP')['PTS'].mean()
    by_opponent['median'] = log.groupby('MATCHUP')['PTS'].median()
    return by_opponent

def make_day_table(log):
    log = log[['GAME_DATE','WL']]
    log['GAME_DATE'] = pd.to_datetime(log['GAME_DATE'], format="%b %d, %Y")
    log['day_in_month'] = log['GAME_DATE'].map(lambda x: int(x.strftime("%d")))
    log['day_in_week'] = log['GAME_DATE'].map(lambda x: int(x.strftime("%w")))
    return log

def make_subplot(log_1,log_2,team_1,team_2,by,color=None):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(team_1,team_2),
        shared_yaxes=True)

    fig.append_trace(
        go.Scatter(
            x=np.arange(log_1[by].count()), 
            y=log_1[by],
            name=f'{team_1} points'),
        row=1, col=1       
    )

    fig.append_trace(
        go.Scatter(
            x=np.arange(log_2[by].count()), 
            y=log_2[by],
            name=f'{team_2} points'),
        row=1, col=2      
    )

    #Regression model

        #team 1
    y = log_1['PTS']
    #training data
    X = np.arange(y.count())[:,np.newaxis]
    #model
    model = LinearRegression().fit(X,y)
    
    #new predictors
    xfit = np.arange(y.count()+5)
    Xfit = xfit[:, np.newaxis]
    #predictand values
    yfit = model.predict(Xfit)

        # team 2
    y_2 = log_2['PTS']
    #training data
    X_2 = np.arange(y_2.count())[:,np.newaxis]
    #model
    model_2 = LinearRegression().fit(X_2,y_2)
    
    #new predictors
    xfit_2 = np.arange(y_2.count()+5)
    Xfit_2 = xfit_2[:, np.newaxis]
    #predictand values
    yfit_2 = model_2.predict(Xfit_2)

    #lines
    fig.append_trace(
        go.Scatter(
            x=xfit, 
            y=yfit,
            name=f'{team_1} points trend'),
        row=1, col=1
    )

    fig.append_trace(
        go.Scatter(
            x=xfit_2, 
            y=yfit_2,
            name=f'{team_2} points trend'),
        row=1, col=2
    )

    fig.update_layout(title_text=f'Comparasion of {team_1} and {team_2} scored points from this season.')
    fig.show()

# #TODO kibővíteni azzal hogy mindkét ploton látszódjon az adott csapat által szerzett, és kapott pontok boxplot-ja
# https://stackoverflow.com/questions/55698429/different-box-plot-series-traces-within-plotly-subplots
def make_hist_subplot(log_1,log_2,team_1,team_2,by):   
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(team_1,team_2),
        shared_yaxes=True)

    fig.add_trace(
        go.Box(
            y=log_1[by],
            name=f'{team_1} points'),
        row=1, col=1
    )

    fig.add_trace(
        go.Box(
            y=log_2[by],
            name=f'{team_2} points'),
        row=1, col=2
    )

    fig.update_layout(title_text=f'Comparasion of {team_1} and {team_2} scored points from this season with boxplot.')
    fig.show()