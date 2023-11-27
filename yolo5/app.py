import time
from pathlib import Path, PosixPath

from botocore.exceptions import ClientError
from flask import Flask, request, jsonify
from detect import run
import uuid
import yaml
from loguru import logger
from dotenv import load_dotenv
import os
import boto3
from pymongo import MongoClient
import json
from bson import ObjectId

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        elif isinstance(o, PosixPath):
            return str(o)
        return json.JSONEncoder.default(self, o)

aws_key_id = os.getenv('AWS_KEY_ID')
aws_access_key = os.getenv('AWS_ACCESS_KEY')
region = os.getenv('REGION')
session = boto3.Session(aws_access_key_id=aws_key_id, aws_secret_access_key=aws_access_key, region_name=region)

images_bucket = os.getenv('BUCKET_NAME')
mongo_uri = os.getenv('MONGO_URI')
mongo_client = MongoClient(mongo_uri)
db = mongo_client['docker-yolo-db']
collection = db['predicted-summaries']

with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']

app = Flask(__name__)
app.json_encoder = JSONEncoder


@app.route('/predict', methods=['POST'])
def predict():
    # Generates a UUID for this current prediction HTTP request. This id can be used as a reference in logs to identify and track individual prediction requests.
    prediction_id = str(uuid.uuid4())

    logger.info(f'prediction: {prediction_id}. start processing')

    # Receives a URL parameter representing the image to download from S3
    img_name = request.args.get('imgName')

    # TODO download img_name from S3, store the local image path in original_img_path
    #  The bucket name should be provided as an env var BUCKET_NAME.
    s3 = session.client('s3')
    original_img_path = img_name
    if "photos" in img_name:
        os.makedirs("photos", exist_ok=True)
        img_name = img_name.split('/')[-1]
    try:
        s3.download_file(images_bucket, original_img_path, original_img_path)
        logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')
    except Exception as e:
        logger.error(f'Error downloading image from S3:{e}')
        return 'Error downloading image from S3'
    # Predicts the objects in the image
    run(
        weights='yolov5s.pt',
        data='data/coco128.yaml',
        source=original_img_path,
        project='static/data',
        name=prediction_id,
        save_txt=True
    )

    logger.info(f'prediction: {prediction_id}/{original_img_path}. done')

    # This is the path for the predicted image with labels
    # The predicted image typically includes bounding boxes drawn around the detected objects, along with class labels and possibly confidence scores.
    predicted_img_path = Path(f'static/data/{prediction_id}/{img_name}')

    # TODO Uploads the predicted image (predicted_img_path) to S3 (be careful not to override the original image).
    upload_file_to_s3(str(predicted_img_path), images_bucket, f"predicted/{img_name}")
    # Parse prediction labels and create a summary
    pred_summary_path = Path(f'static/data/{prediction_id}/labels/{img_name.split(".")[0]}.txt')
    if pred_summary_path.exists():
        with open(pred_summary_path) as f:
            labels = f.read().splitlines()
            labels = [line.split(' ') for line in labels]
            labels = [{
                'class': names[int(l[0])],
                'cx': float(l[1]),
                'cy': float(l[2]),
                'width': float(l[3]),
                'height': float(l[4]),
            } for l in labels]

        logger.info(f'prediction: {prediction_id}/{original_img_path}. prediction summary:\n\n{labels}')

        prediction_summary = {
            'prediction_id': prediction_id,
            'original_img_path': original_img_path,
            'predicted_img_path': str(predicted_img_path),
            'labels': labels,
            'time': time.time()
        }
        # TODO store the prediction_summary in MongoDB
        save_to_db(prediction_summary)
        prediction_summary["_id"] = str(prediction_summary["_id"])
        return jsonify(prediction_summary)
    else:
        return f'prediction: {prediction_id}/{original_img_path}. prediction result not found', 404


# a function to upload predicted image to S3
def upload_file_to_s3(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = file_name
    s3_client = session.client('s3')
    try:
        s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logger.error(e)
        return False
    return True


# a function that saves a file to the db collection
def save_to_db(data):
    try:
        logger.info("Saving Data to Collection..")
        collection.insert_one(data)
        logger.info("Saved Data Successfully")
        return True
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        return False


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)
