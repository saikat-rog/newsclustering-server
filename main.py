from app.services.config import create_app
import os

app = create_app()

@app.route("/")
def welcome():
    return "bro this is fun of news!"

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)