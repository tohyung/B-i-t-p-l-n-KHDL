import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, movies_path, tags_path=None):
        self.movies_path = movies_path
        self.tags_path = tags_path

        self.movies = None
        self.tags = None
        self.movie_features = None
        self.tfidf_matrix = None
        self.vectorizer = None

        self.load_data()
        self.preprocess_data()
        self.build_model()

    def load_data(self):
        self.movies = pd.read_csv(self.movies_path)

        if self.tags_path is not None:
            self.tags = pd.read_csv(self.tags_path)
        else:
            self.tags = None

    def clean_text(self, text):
        if pd.isna(text):
            return ""

        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def preprocess_data(self):
        self.movie_features = self.movies.copy()

        self.movie_features["genres_clean"] = (
            self.movie_features["genres"]
            .fillna("")
            .str.replace("|", " ", regex=False)
            .str.lower()
        )

        if self.tags is not None:
            self.tags["tag_clean"] = self.tags["tag"].apply(self.clean_text)

            tags_grouped = (
                self.tags
                .groupby("movieId")["tag_clean"]
                .apply(lambda x: " ".join(x))
                .reset_index()
            )

            tags_grouped.rename(columns={"tag_clean": "tags_clean"}, inplace=True)

            self.movie_features = self.movie_features.merge(
                tags_grouped,
                on="movieId",
                how="left"
            )

            self.movie_features["tags_clean"] = self.movie_features["tags_clean"].fillna("")
        else:
            self.movie_features["tags_clean"] = ""

        # Lặp genres 3 lần để genres có trọng số mạnh hơn tags
        self.movie_features["content"] = (
            self.movie_features["genres_clean"] + " " +
            self.movie_features["genres_clean"] + " " +
            self.movie_features["genres_clean"] + " " +
            self.movie_features["tags_clean"]
        )

    def build_model(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=1
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.movie_features["content"]
        )

    def find_movie(self, movie_title):
        movie_title = movie_title.lower().strip()

        exact_match = self.movie_features[
            self.movie_features["title"].str.lower() == movie_title
        ]

        if not exact_match.empty:
            return exact_match.iloc[0]

        contains_match = self.movie_features[
            self.movie_features["title"].str.lower().str.contains(movie_title, na=False)
        ]

        if not contains_match.empty:
            return contains_match.iloc[0]

        return None

    def get_movie_suggestions(self, movie_title, limit=5):
        movie_title = movie_title.lower().strip()

        suggestions = self.movie_features[
            self.movie_features["title"].str.lower().str.contains(movie_title, na=False)
        ][["movieId", "title", "genres"]].head(limit)

        return suggestions

    def recommend(self, movie_title, top_n=10):
        movie = self.find_movie(movie_title)

        if movie is None:
            suggestions = self.get_movie_suggestions(movie_title)

            return {
                "status": "not_found",
                "message": "Movie not found.",
                "suggestions": suggestions
            }

        movie_index = movie.name

        similarity_scores = cosine_similarity(
            self.tfidf_matrix[movie_index],
            self.tfidf_matrix
        ).flatten()

        similar_indices = similarity_scores.argsort()[::-1]

        recommendations = []

        for idx in similar_indices:
            if idx == movie_index:
                continue

            recommendations.append({
                "movieId": self.movie_features.iloc[idx]["movieId"],
                "title": self.movie_features.iloc[idx]["title"],
                "genres": self.movie_features.iloc[idx]["genres"],
                "similarity": similarity_scores[idx]
            })

            if len(recommendations) == top_n:
                break

        result = pd.DataFrame(recommendations)

        return {
            "status": "success",
            "input_movie": {
                "movieId": movie["movieId"],
                "title": movie["title"],
                "genres": movie["genres"]
            },
            "recommendations": result
        }