from main import app

# Restarts Gunicorn server on save when True
app.config['DEBUG'] = True

if __name__ == "__main__":
    app.run()
