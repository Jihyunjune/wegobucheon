from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from typing import List, Dict
from dataclasses import dataclass
from flask_cors import CORS

app = Flask(__name__)
#CORS(app, resources={r"/*": {"origins": "*"}})  # 모든 출처에 대해 CORS 허용
CORS(app, resources={r"/*": {"origins": "https://wegobucheon.netlify.app"}})

tourist_spots = [
    {"name": "부천식물원", "category": "산책 및 자연", "description": "다양한 식물 전시와 자연 체험이 가능한 정원", "latitude": 37.50512276, "longitude": 126.8157628, "url": "https://www.bucheon.go.kr/site/homepage/menu/viewMenu?menuid=148006001004010"},
    {"name": "플레이아쿠아리움", "category": "동물 및 생태 체험", "description": "다양한 해양 생물과 수족관 관람 가능", "latitude": 37.49942786, "longitude": 126.7440795, "url": "https://map.naver.com/v5/entry/place/1820740985"},
    {"name": "볼베어파크", "category": "어린이 및 가족 체험", "description": "실내 놀이 및 어드벤처 체험 공간", "latitude": 37.49942786, "longitude": 126.7440795, "url": "https://map.naver.com/v5/entry/place/1654560840"},
    # 더 많은 관광지 추가...
] 

@dataclass
class TourSpot:
    name: str
    category: str
    rating: float
    reviews: List[str]
    location: tuple  # (latitude, longitude)
    features: List[str]  # 관광지의 특징들

    def calculate_sentiment_score(self):
        positive_words = ['좋', '최고', '추천', '훌륭', '만족']
        return len([r for r in self.reviews if any(word in r for word in positive_words)]) / len(
            self.reviews) if self.reviews else 0

class GLocalKernelLayer(tf.keras.layers.Layer):
    def __init__(self, n_hidden, n_dim=5, **kwargs):
        super(GLocalKernelLayer, self).__init__(**kwargs)
        self.n_hidden = n_hidden
        self.n_dim = n_dim

    def build(self, input_shape):
        self.W = self.add_weight(
            name='W',
            shape=[input_shape[-1], self.n_hidden],
            initializer='glorot_uniform',
            trainable=True
        )
        self.u = self.add_weight(
            name='u',
            shape=[input_shape[-1], 1, self.n_dim],
            initializer='truncated_normal',
            trainable=True
        )
        self.v = self.add_weight(
            name='v',
            shape=[1, self.n_hidden, self.n_dim],
            initializer='truncated_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='b',
            shape=[self.n_hidden],
            initializer='zeros',
            trainable=True
        )
        super(GLocalKernelLayer, self).build(input_shape)

    def call(self, x):
        # Local kernel computation
        dist = tf.norm(self.u - self.v, ord=2, axis=2)
        w_hat = tf.maximum(0., 1. - dist ** 2)

        # Effective weight matrix
        W_eff = self.W * w_hat
        return tf.nn.sigmoid(tf.matmul(x, W_eff) + self.b)

class GLocalTourismRecommender:
    def __init__(self, n_hidden=500, n_dim=5, n_layers=2):
        self.n_hidden = n_hidden
        self.n_dim = n_dim
        self.n_layers = n_layers

        # 부천시 관광지 데이터 초기화
        self.spots = {
            "상동호수공원": TourSpot(
                "상동호수공원",
                "산책/자연",
                4.5,
                ["로봇과 체험이 너무 재미있어요", "아이들이 좋아해요", "교육적이에요"],
                (37.5044, 126.7642),
                ["공원","호수","산책","놀이터","운동","아이","식물원","주차"]
            ),
            "도당수목원": TourSpot(
                "도당수목원",
                "산책/자연",
                4.3,
                ["전시가 멋져요", "카페도 좋아요", "사진찍기 좋아요"],
                (37.4855, 126.7824),
                ["장미","공원","산책","축제","백만송이","사진","걷기"]
            ),
            "무릉도원수목원": TourSpot(
                "무릉도원수목원",
                "산책/자연",
                4.2,
                ["산책하기 좋아요", "조용하고 평화로워요", "주차가 편해요"],
                (37.5123, 126.7890),
                ["산책","체험","식물원","가을","산책로","수목원","걷기","공원","튤립"]
            ),
            "백만송이장미원": TourSpot(
                "백만송이장미원",
                "산책/자연",
                4.0,
                ["시설이 좋아요", "운동하기 좋아요", "깨끗해요"],
                (37.5033, 126.7756),
                ["장미","공원","주차","축제","사진","구경","산책","백만송이"]
            ),
            "원미산진달래공원": TourSpot(
                "원미산진달래공원",
                "산책/자연",
                4.0,
                ["시설이 좋아요", "운동하기 좋아요", "깨끗해요"],
                (37.5033, 126.7756),
                ['진달래','벚꽃','축제','구경','운동장','사진','산책']
            ),
            "플레이아쿠아리움": TourSpot(
                "플레이아쿠아리움",
                "동물/생태체험",
                4.0,
                ["시설이 좋아요", "운동하기 좋아요", "깨끗해요"],
                (37.5033, 126.7756),
                ['아이','동물','아쿠아리움','물고기','호랑이','인어공주','구경']
            ),
            "나눔공장": TourSpot(
                "나눔공장",
                "동물/생태체험",
                4.0,
                ["시설이 좋아요", "운동하기 좋아요", "깨끗해요"],
                (37.5033, 126.7756),
                ['아이','동물','체험','먹이','염소','토끼','우유']
            ),
            "부천자연생태공원": TourSpot(
                "부천자연생태공원",
                "동물/생태체험",
                4.0,
                ["시설이 좋아요", "운동하기 좋아요", "깨끗해요"],
                (37.5033, 126.7756),
                ['정원','산책','아이','식물원','튤립','생태공원','구경','박물관','나들이']
            ),
            "볼베어파크": TourSpot(
                "플레이아쿠아리움",
                "어린이/가족",
                4.0,
                ["시설이 좋아요", "운동하기 좋아요", "깨끗해요"],
                (37.5033, 126.7756),
                ['아이','규모','카페','파크','키즈','썰매','좋은시설']
            ),
            "부천로보파크": TourSpot(
                "부천로보파크",
                "어린이/가족",
                4.0,
                ["시설이 좋아요", "운동하기 좋아요", "깨끗해요"],
                (37.5033, 126.7756),
                ['아이','체험','로봇','관람','전시','투어']
            ),
            "부천아트벙커": TourSpot(
                "부천아트벙커",
                "문화/예술",
                4.0,
                ["시설이 좋아요", "운동하기 좋아요", "깨끗해요"],
                (37.5033, 126.7756),
                ['전시','카페','음식','분위기','문화','예술','작품','파스타']
            ),
            "부천아트센터": TourSpot(
                "부천아트센터",
                "문화/예술",
                4.0,
                ["시설이 좋아요", "운동하기 좋아요", "깨끗해요"],
                (37.5033, 126.7756),
                ['공연','음향','관람','아트','연주','카페','주차']
            ),
            "레노부르크뮤지엄": TourSpot(
                "레노부르크뮤지엄",
                "문화/예술",
                4.0,
                ["시설이 좋아요", "운동하기 좋아요", "깨끗해요"],
                (37.5033, 126.7756),
                ['아이','카페','사진','브런치','전시','관람','미디어아트']
            ),
            "부천시립박물관": TourSpot(
                "부천시립박물관",
                "전통/역사",
                4.0,
                ["시설이 좋아요", "운동하기 좋아요", "깨끗해요"],
                (37.5033, 126.7756),
                ['박물관','아이','옹기','전시','체험','관람','교육']
            ),
            "고강선사유적공원": TourSpot(
                "고강선사유적공원",
                "전통/역사",
                4.0,
                ["시설이 좋아요", "운동하기 좋아요", "깨끗해요"],
                (37.5033, 126.7756),
                ['공원','산책','선사','철쭉','유적','도서관','운동','아이']
            ),
            "부천한옥체험마을": TourSpot(
                "부천한옥체험마을",
                "전통/역사",
                4.0,
                ["시설이 좋아요", "운동하기 좋아요", "깨끗해요"],
                (37.5033, 126.7756),
                ['전통','마을','숙박','만화','박물관','사진']
            ),
            "부천천문과학관": TourSpot(
                "부천천문과학관",
                "과학/교육",
                4.0,
                ["시설이 좋아요", "운동하기 좋아요", "깨끗해요"],
                (37.5033, 126.7756),
                ['아이','관측','태양','별자리','체험','과학관']
            ),
            "한국만화박물관": TourSpot(
                "한국만화박물관",
                "과학/교육",
                4.0,
                ["시설이 좋아요", "운동하기 좋아요", "깨끗해요"],
                (37.5033, 126.7756),
                ['아이','만화','체험','전시','박물관','추억','도서관']
            )
        }
        self._build_rating_matrix()
        self._initialize_model()

    def _build_rating_matrix(self):
        """사용자-관광지 평점 행렬 생성"""
        self.n_spots = len(self.spots)
        self.n_users = 100  # 예시 사용자 수
        self.rating_matrix = np.zeros((self.n_spots, self.n_users), dtype=np.float32)

    def _initialize_model(self):
        """Keras 모델 초기화"""
        inputs = tf.keras.Input(shape=(self.n_users,))
        x = inputs

        # GLocal 레이어 추가
        for i in range(self.n_layers):
            x = GLocalKernelLayer(self.n_hidden, name=f'glocal_layer_{i}')(x)

        # 최종 출력 레이어
        outputs = GLocalKernelLayer(self.n_users, name='output_layer')(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

    def calculate_spot_score(self, spot: TourSpot, user_interests: List[str]) -> float:
        interest_score = 1.0
        if any(interest in spot.features for interest in user_interests):
            interest_score = 1.5
        sentiment_score = spot.calculate_sentiment_score()
        final_score = (spot.rating * 0.4) + (sentiment_score * 0.3) + (interest_score * 0.3)
        return final_score

    def recommend_spots(self, user_interests: List[str], num_recommendations: int = 3) -> List[Dict]:
        """사용자 맞춤 관광지 추천"""
        spot_scores = []
        for spot_name, spot in self.spots.items():
            score = self.calculate_spot_score(spot, user_interests)
            spot_scores.append({
                'name': spot.name,
                'category': spot.category,
                'rating': spot.rating,
                'score': score,
                'reviews': spot.reviews[:2],
                'location': spot.location,
                'features': spot.features
            })

        # 모델 예측을 통한 점수 보정
        predictions = self.model.predict(self.rating_matrix, verbose=0)

        # 예측 점수를 반영하여 최종 점수 조정
        for i, spot_score in enumerate(spot_scores):
            model_score = float(np.mean(predictions[i]))  # float 변환 추가
            spot_score['score'] = spot_score['score'] * 0.7 + model_score * 0.3

        spot_scores.sort(key=lambda x: x['score'], reverse=True)
        return spot_scores[:num_recommendations]

# 추천기 인스턴스 생성
recommender = GLocalTourismRecommender()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    category = request.args.get('category')
    filtered_spots = [spot for spot in tourist_spots if spot['category'] == category]
    return render_template('result.html', category=category, spots=filtered_spots)

@app.route('/ai_match')
def ai_match():
    return render_template('ai_match.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_interests = request.form.getlist('interests')
    recommendations = recommender.recommend_spots(user_interests)
    return jsonify(recommendations)

# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)