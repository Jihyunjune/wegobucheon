from flask import Flask, request, jsonify, render_template, render_template_string
import numpy as np
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
import sqlite3
import ast
from collections import Counter
import urllib
from urllib.request import Request, urlopen
import folium
import json
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

# 네이버 API 키 설정
client_id = '0uxifb6sxl'
client_secret = 'h9VLdh7W1xOzIwf2RtUxXMH9X8b0Y779kytjgNaq'

@dataclass
class TourSpot:
    name: str
    category: str
    rating: float
    reviews: List[str]
    location: tuple  # (latitude, longitude)
    features: List[str]
    address: str = ''

    def calculate_sentiment_score(self, analyzer):
        """KoNLPy 기반 감성 점수 계산"""
        if not self.reviews:
            return 0.0

        if analyzer:
            sentiment_analysis = analyzer.process_reviews(self.reviews)
            return sentiment_analysis['sentiment_score']
        return 0.5

class KoNLPySentimentAnalyzer:
    def __init__(self):
        self.okt = Okt()
        self.sentiment_words = {
            'positive': [
                '좋다', '최고', '훌륭하다', '추천', '만족', '친절하다', '깨끗하다',
                '재미있다', '즐겁다', '멋지다', '편리하다', '괜찮다', '예쁘다',
                '편하다', '신나다', '흥미롭다', '특별하다', '인상적', '감동',
                '맛있다', '즐기다', '행복하다', '안전하다', '매력적'
            ],
            'negative': [
                '별로', '실망', '불만', '나쁘다', '후회', '불친절하다', '더럽다',
                '비싸다', '좁다', '복잡하다', '불편하다', '아쉽다', '부족하다',
                '지루하다', '심심하다', '재미없다', '위험하다', '불안하다',
                '시끄럽다', '힘들다', '불쾌하다', '싫다', '답답하다'
            ]
        }

    def analyze_review(self, text: str) -> Dict:
        morphs = self.okt.pos(text, norm=True, stem=True)
        positive_score = 0
        negative_score = 0
        nouns = []

        for word, tag in morphs:
            weight = 1.2 if tag in ['Adjective', 'Verb'] else 1.0
            if word in self.sentiment_words['positive']:
                positive_score += weight
            elif word in self.sentiment_words['negative']:
                negative_score += weight

            if word in ['않다', '안', '못', '없다']:
                positive_score, negative_score = negative_score, positive_score

            if tag in ['Noun', 'NNG', 'NNP']:
                nouns.append(word)

        total_score = positive_score + negative_score
        sentiment_score = positive_score / total_score if total_score > 0 else 0.5

        return {
            'sentiment_score': sentiment_score,
            'nouns': nouns,
            'positive_score': positive_score,
            'negative_score': negative_score
        }

    def process_reviews(self, reviews: List[str]) -> Dict:
        review_analyses = [self.analyze_review(review) for review in reviews]
        avg_sentiment = np.mean([analysis['sentiment_score'] for analysis in review_analyses])
        all_nouns = [word for analysis in review_analyses for word in analysis['nouns']]
        top_keywords = [word for word, _ in Counter(all_nouns).most_common(10)]

        sentiment_distribution = {
            'positive': len([a for a in review_analyses if a['sentiment_score'] > 0.6]),
            'negative': len([a for a in review_analyses if a['sentiment_score'] < 0.4]),
            'neutral': len([a for a in review_analyses if 0.4 <= a['sentiment_score'] <= 0.6])
        }

        return {
            'sentiment_score': avg_sentiment,
            'keywords': top_keywords,
            'analysis': sentiment_distribution
        }

class TourismRecommender:
    def __init__(self, db_path: str = 'tourism.db'):
        self.setup_logging()
        self.spots = {}
        self.db_path = db_path
        self.sentiment_analyzer = KoNLPySentimentAnalyzer()

        # TF-IDF 벡터라이저 초기화
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.95,
            token_pattern=r'[가-힣]+|[a-zA-Z]+'
        )

        # 데이터베이스 초기화 및 CSV 로드
        self._initialize_database()
        self.load_spots_from_csv('tourism_spots.csv')
    def _initialize_database(self):
        """데이터베이스 테이블 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS spots (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    rating FLOAT,
                    location_lat FLOAT,
                    location_lon FLOAT,
                    features TEXT,
                    created_at TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY,
                    spot_id INTEGER,
                    content TEXT,
                    sentiment_score FLOAT,
                    created_at TIMESTAMP,
                    FOREIGN KEY (spot_id) REFERENCES spots (id)
                )
            """)   
    def setup_logging(self):
        self.logger = logging.getLogger('TourismRecommender')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def load_spots_from_csv(self, csv_path: str):
        try:
            if not Path(csv_path).exists():
                self.logger.error(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
                return False

            df = pd.read_csv(csv_path, encoding="utf-8")

            for _, row in df.iterrows():
                try:
                    reviews = ast.literal_eval(row['reviews'])
                    features = ast.literal_eval(row['features'])
                    location = ast.literal_eval(row['location'])
                    address = row['address']
                except:
                    self.logger.error(f"데이터 파싱 오류 발생: {row['name']}")
                    continue

                spot = TourSpot(
                    name=row['name'],
                    category=row['category'],
                    rating=float(row['rating']),
                    reviews=reviews,
                    location=location,
                    features=features,
                    address=address
                )

                self.spots[row['name']] = spot

            print(f"총 {len(self.spots)}개의 관광지 데이터를 로드했습니다.")
            return True

        except Exception as e:
            self.logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
            return False

    def recommend_spots(self, user_interests: List[str], num_recommendations: int = 3) -> List[Dict]:
        spot_scores = []
        for spot_name, spot in self.spots.items():
            base_score = self.calculate_spot_score(spot, user_interests)
            review_analysis = self.sentiment_analyzer.process_reviews(spot.reviews)
            final_score = (base_score * 0.6) + (review_analysis['sentiment_score'] * 0.4)
            spot_scores.append({
                'name': spot_name,
                'category': spot.category,
                'rating': spot.rating,
                'features': spot.features,
                'address': spot.address,
                'location': spot.location,
                'final_score': final_score
            })
        spot_scores.sort(key=lambda x: x['final_score'], reverse=True)
        return spot_scores[:num_recommendations]

    def calculate_spot_score(self, spot: TourSpot, user_interests: List[str]) -> float:
        interest_score = 1.5 if any(interest in spot.features for interest in user_interests) else 1.0
        return (spot.rating + interest_score) / 2

# 네이버 지도 API를 사용해 주소를 좌표로 변환하는 함수
def get_location(loc):
    url = f"https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode?query=" + urllib.parse.quote(loc)
    request = Request(url)
    request.add_header('X-NCP-APIGW-API-KEY-ID', client_id)
    request.add_header('X-NCP-APIGW-API-KEY', client_secret)
    response = urlopen(request)
    if response.getcode() == 200:
        response_body = json.loads(response.read().decode('utf-8'))
        if response_body['meta']['totalCount'] == 1:
            lat = response_body['addresses'][0]['y']
            lon = response_body['addresses'][0]['x']
            return (lon, lat)
    return None

# 인도(도보) 경로를 얻는 함수
def get_optimal_route(start, goal, waypoints=None, option=''):
    waypoints_str = '|'.join([f"{wp[0]},{wp[1]}" for wp in waypoints]) if waypoints else ''
    url = f'https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving?start={start[0]},{start[1]}&goal={goal[0]},{goal[1]}&waypoints={waypoints_str}&option={option}'
    request = Request(url)
    request.add_header('X-NCP-APIGW-API-KEY-ID', client_id)
    request.add_header('X-NCP-APIGW-API-KEY', client_secret)
    response = urlopen(request)
    if response.getcode() == 200:
        return json.loads(response.read().decode('utf-8'))
    return None


# 경로를 시각화하고 자동 확대 조정하는 함수
def visualize_route(route_data):
    start = (route_data['route']['traoptimal'][0]['summary']['start']['location'][1],
             route_data['route']['traoptimal'][0]['summary']['start']['location'][0])
    goal = (route_data['route']['traoptimal'][0]['summary']['goal']['location'][1],
            route_data['route']['traoptimal'][0]['summary']['goal']['location'][0])

    route_map = folium.Map(location=start, zoom_start=14)
    path_coordinates = [(point[1], point[0]) for point in route_data['route']['traoptimal'][0]['path']]
    route_map.fit_bounds([[min(lat for lat, lon in path_coordinates), min(lon for lat, lon in path_coordinates)],
                          [max(lat for lat, lon in path_coordinates), max(lon for lat, lon in path_coordinates)]])

    # 출발지 마커
    folium.Marker(
        start,
        popup='출발지',
        icon=folium.Icon(color='green', icon='home', prefix='fa')
    ).add_to(route_map)

        # 경유지 마커 추가 (원형 숫자 마커)
    waypoints = route_data['route']['traoptimal'][0]['summary'].get('waypoints', [])
    for i, waypoint in enumerate(waypoints):
        location = (waypoint['location'][1], waypoint['location'][0])
        folium.Marker(
            location,
            popup=f'경유지 {i + 1}',
            icon=folium.DivIcon(
                icon_size=(30, 30),
                icon_anchor=(15, 15),
                html=f'<div style="font-size: 12px; color: white; background-color: orange; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center;">{i + 1}</div>'
            )
        ).add_to(route_map)

    # 도착지 마커
    folium.Marker(
        goal,
        popup='도착지',
        icon=folium.Icon(color='red', icon='flag', prefix='fa')
    ).add_to(route_map)

    # 경로 라인 추가
    folium.PolyLine(path_coordinates, color="blue", weight=5, opacity=0.7).add_to(route_map)

    return route_map

recommender = TourismRecommender()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    category = request.args.get('category')
    filtered_spots = [
        {
            'name': spot.name,
            'category': spot.category,
            'description': spot.reviews[0] if spot.reviews else "",
            'latitude': spot.location[0],
            'longitude': spot.location[1]
        } 
        for spot in recommender.spots.values() if spot.category == category
    ]
    return render_template('result.html', category=category, spots=filtered_spots)

@app.route('/ai_match')
def ai_match():
    return render_template('ai_match.html')

@app.route('/ai_map', methods=['GET', 'POST'])
def ai_map():
    if request.method == 'POST':
        selected_spots = request.json.get("selectedSpots", [])
        return render_template('ai_map.html', selected_spots=selected_spots)
    else:
        return render_template('ai_map.html', selected_spots=[])

@app.route('/recommend', methods=['POST'])
def recommend():
    user_interests = request.json.get('interests', [])
    recommendations = recommender.recommend_spots(user_interests)
    return jsonify(recommendations)

@app.route('/ai_map', methods=['GET', 'POST'], endpoint='ai_map_page')
def ai_map():
    if request.method == 'POST':
        selected_spots = request.json.get("selectedSpots", [])
        return render_template('ai_map.html', selected_spots=selected_spots)
    else:
        return render_template('ai_map.html', selected_spots=[])

@app.route('/ai_map/show_route', methods=['POST'], endpoint='show_route_on_ai_map')
@app.route('/show_route', methods=['POST'])
def show_route():
    start_address = request.form['start']
    goal_address = request.form['goal']
    waypoint_addresses = request.form.getlist('waypoint')

    start = get_location(start_address)
    goal = get_location(goal_address)
    waypoints = [get_location(addr) for addr in waypoint_addresses if addr]

    if not start or not goal:
        return "출발지 또는 도착지 좌표를 찾을 수 없습니다."

    route_data = get_optimal_route(start, goal, waypoints=waypoints)
    if not route_data:
        return "경로를 찾을 수 없습니다."

    route_map = visualize_route(route_data)
    route_map.save('templates/route_map.html')  # 'route_map.html' 파일을 templates 폴더에 저장

    # route_map.html 템플릿을 렌더링하여 반환
    return render_template('route_map.html')


if __name__ == '__main__':
    app.run(debug=True)
