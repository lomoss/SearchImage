from app import app


def main():
    app.run(host='127.0.0.1', port=5000, threaded=True)

if __name__ == '__main__':
    main()