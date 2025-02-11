from flask import Flask
import views


app = Flask('__name__') #webserver gateway interphase 


app.add_url_rule(rule='/', endpoint='home', view_func=views.index, methods=['GET', 'POST'])

if __name__ == "__main__":
    app.run(debug=True)