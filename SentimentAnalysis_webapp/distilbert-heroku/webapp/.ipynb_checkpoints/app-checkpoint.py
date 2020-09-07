# Setup
import os

import dash
import dash_bootstrap_components as dbc
import dash_daq as daq
import dash_html_components as html
from dash.dependencies import Input, Output, State

from python.nlp import get_probability

BS = "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css"
app = dash.Dash(__name__,
                external_stylesheets=[BS],
                assets_folder=os.getcwd() + '/webapp/static')


navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("IMDb Dataset", href="https://ai.stanford.edu/~amaas/data/sentiment/")),
        dbc.NavItem(dbc.NavLink("DistilBERT", href="https://huggingface.co/transformers/model_doc/distilbert.html")),
    ],
    brand="Sentiment Analysis Webapp",
    brand_href="#",
    className="navbar navbar-expand-lg navbar-dark bg-dark sticky-top",
    fluid=True
)



app.layout = html.Div([
    navbar,
    dbc.Row([
        dbc.Col(
                html.Div([
                    html.A([
                        html.Img(src=app.get_asset_url("linkedinlogo.png"),
                                  className="img-fluid mb-4 ml-2",
                                 )
                        ],
                        href='https://www.linkedin.com/in/lschneidpro'
                        ),
                    html.A(
                        [html.Img(src=app.get_asset_url("githublogo.png"),
                                  className="img-fluid ml-2",
                                  )
                         ],
                        href='https://github.com/lschneidpro'),
                    ],
                    className="mt-3"
                    ),
                className="col-1 border-right"),
        
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H2("Sentiment Analysis of Movie Reviews"),
                        html.H6("Sentiment score from DistilBERT fine-tuned on IMDb Dataset",
                                className="text-muted"
                                ),
                        ],
                        className="mt-3"),
                    ],
                    ),
                ],
                ),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Textarea(id='text_input', bs_size="lg", placeholder="Enter review and click analyze"),
                        html.Br(),
                        dbc.Button("Analyze", id="button", className="mr-2"),
                        html.Br(),
                        daq.Gauge(id='gauge',
                            color={"gradient": True,
                                   "ranges": {"red": [0, 33],
                                              "yellow": [33, 63],
                                              "green": [63, 100]}
                                   },
                            max=100,
                            min=0,
                            units='%',
                            value=0,
                            ),
                        html.H6(id='text_output',
                                children='',
                                className="text-muted",
                                ),
                        ],
                        className="container mt-5 text-center"),
                    ],
                    ),
                ],
                ),
            ],
            ),
        ],
        ),
    ],
    )

@app.callback(
    [Output('gauge','value'),
     Output('text_output','children')],
    [Input('button','n_clicks')],
    [State('text_input', 'value')])
def update_output(n_clicks, text):
    
    if n_clicks is None:
        return 0, ''
        
    else:
        prob = get_probability(text)
        if prob < 0.5:
            return prob*100, 'Negative with confidence of {:0.0f} %'.format((1-prob)*100)
        else:
            return prob*100, 'Positive with confidence of {:0.0f} %'.format(prob*100)
    

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run_server(host='0.0.0.0', port=port)







