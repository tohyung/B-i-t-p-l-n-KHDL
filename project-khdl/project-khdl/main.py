from src.content_based import ContentBasedRecommender


def print_recommendations(response):
    if response["status"] == "not_found":
        print("\nMovie not found.")

        suggestions = response["suggestions"]

        if suggestions.empty:
            print("No similar movie title found.")
        else:
            print("\nDid you mean:")
            for row in suggestions.itertuples(index=False):
                print(f"- {row.title} | Movie ID: {row.movieId} | Genres: {row.genres}")

        return

    input_movie = response["input_movie"]
    recommendations = response["recommendations"]

    print("\n" + "=" * 90)
    print("CONTENT-BASED RECOMMENDATION")
    print("Based on: Genres + Tags")
    print("=" * 90)

    print(f"\nInput Movie:")
    print(f"Movie ID : {input_movie['movieId']}")
    print(f"Title    : {input_movie['title']}")
    print(f"Genres   : {input_movie['genres']}")

    print("\nRecommended Movies:\n")

    print(f"{'Rank':<6} {'Movie ID':<10} {'Title':<45} {'Similarity':<10}")
    print("-" * 90)

    for rank, row in enumerate(recommendations.itertuples(index=False), start=1):
        title = row.title

        if len(title) > 43:
            title = title[:40] + "..."

        print(
            f"{rank:<6} "
            f"{row.movieId:<10} "
            f"{title:<45} "
            f"{row.similarity:.4f}"
        )


def main():
    movies_path = "data/movies.csv"
    tags_path = "data/tags.csv"

    recommender = ContentBasedRecommender(
        movies_path=movies_path,
        tags_path=tags_path
    )

    while True:
        print("\n" + "=" * 60)
        print("MOVIELENS CONTENT-BASED RECOMMENDATION SYSTEM")
        print("=" * 60)
        print("1. Recommend similar movies")
        print("0. Exit")

        choice = input("\nSelect option: ")

        if choice == "1":
            movie_title = input("Enter movie title: ")
            top_n = input("Enter number of recommendations: ")

            try:
                top_n = int(top_n)
            except ValueError:
                top_n = 10

            response = recommender.recommend(movie_title, top_n)
            print_recommendations(response)

        elif choice == "0":
            print("Exit program.")
            break

        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()