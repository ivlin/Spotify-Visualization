#python3.8.5
from flask import Flask, render_template, send_from_directory
app = Flask(__name__, static_url_path='')

@app.route('/')
def index():
    return app.send_static_file('index.html')


if __name__=="__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=8080, use_reloader=True)
