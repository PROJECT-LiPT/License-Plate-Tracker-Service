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
			id = formData["id"]
			uploader = formData["uploader"]
			data = formData["imgUrl"]

			def data_uri_to_cv2_img(uri):
				encoded_data = uri.split(',')[1]
				nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
				img = cv2.imdecode(nparr, 1)
				return img

			def cv2_img_to_base64(path):
				img = cv2.imread(path)
				_, im_arr = cv2.imencode('.jpg', img)  # im_arr: image in Numpy one-dim array format.
				im_bytes = im_arr.tobytes()
				im_b64 = base64.b64encode(im_bytes)
				return im_b64

			def imgTracker():
				src = data_uri_to_cv2_img(data)
				start = time.time()
				model = E2E()
				image = model.predict(src)
				end = time.time()
				# print('Model process on %.2f s' % (end - start))
				step2_1 = cv2_img_to_base64('./step2_1.png')
				step2_2 = cv2_img_to_base64('./step2_2.png')
				arrayResult = [image,end-start, step2_1, step2_2]
				return arrayResult

			result = []

			if data != '':
				result = imgTracker()
			else :
				result = 'Null Image'

			response = jsonify({
				"id": id,
				"uploader": uploader,
				"imgUrl": data,
				"process": result[1],
				"title": result[0],
				"origin": 'Hồ Chí Minh',
				"step1": result[2].decode('utf-8'),
				"step2": result[3].decode('utf-8')
				})

			response.headers.add('Access-Control-Allow-Origin', '*')

			return response
		except Exception as error:

			return jsonify({
				"statusCode": 500,
				"status": str(error),
				"error": "License Plate Not Found!"
			})