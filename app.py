from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class Simple_Response(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        json_data['key1'] = 'key3'
        return (json_data);

api.add_resource(Simple_Response, '/api')

if __name__ == '__main__':
    app.debug = True
    app.run()