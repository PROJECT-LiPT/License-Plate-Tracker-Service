from flask import Flask, request, jsonify, make_response
import base64
from flask_restx import Api, Resource
import numpy as np
from recognition import E2E # type: ignore
import cv2
import time
from flask_cors import CORS

flask_app = Flask(__name__)
CORS(flask_app)
app = Api(app = flask_app, 
		  version = "1.0", 
		  title = "LicensePlateTracker", 
		  description = "Predict the License Plate Series Number")
name_space = app.namespace('prediction', description='Prediction APIs')

@name_space.route("/")
class MainClass(Resource):
	def options(self):
		response = make_response()
		response.headers.add("Access-Control-Allow-Origin", "*")
		response.headers.add('Access-Control-Allow-Headers', "*")
		response.headers.add('Access-Control-Allow-Methods', "*")
		return response
	def post(self):
		try: 
			formData = request.json
			data = formData["base64"]

			def data_uri_to_cv2_img(uri):
				encoded_data = uri.split(',')[1]
				nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
				img = cv2.imdecode(nparr, 1)
				return img
			
			def imgTracker():
				src = data_uri_to_cv2_img(data)
				start = time.time()
				model = E2E()
				image = model.predict(src)
				end = time.time()
				print('Model process on %.2f s' % (end - start))
				return (image)
			result = ''
			if data != '':
				result = imgTracker()
			else :
				result = 'Null Image'
			response = jsonify({
				"statusCode": 200,
				"status": "Prediction made",
				"result": result
				})
			response.headers.add('Access-Control-Allow-Origin', '*')
			return response
		except Exception as error:
			return jsonify({
				"statusCode": 500,
				"status": "Could not make prediction",
				"error": "License Plate Not Found!"
			})