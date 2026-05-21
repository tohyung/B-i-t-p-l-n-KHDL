-- ============================================================
--  MOVIELENS RECOMMENDATION SYSTEM - DATABASE SCHEMA (MySQL)
--  Người 1: Data + Database + EDA
--  Dataset: MovieLens ml-latest-small
-- ============================================================

CREATE DATABASE IF NOT EXISTS movielens
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE movielens;

-- ------------------------------------------------------------
-- 1. Users
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Users (
    userId      INT          NOT NULL,
    PRIMARY KEY (userId)
);

-- ------------------------------------------------------------
-- 2. Movies
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Movies (
    movieId     INT          NOT NULL,
    title       VARCHAR(500) NOT NULL,
    PRIMARY KEY (movieId)
);

-- ------------------------------------------------------------
-- 3. Genres
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Genres (
    genreId     INT          NOT NULL AUTO_INCREMENT,
    name        VARCHAR(100) NOT NULL UNIQUE,
    PRIMARY KEY (genreId)
);

-- ------------------------------------------------------------
-- 4. MovieGenres  (nhiều-nhiều: Movie <-> Genre)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS MovieGenres (
    movieId     INT          NOT NULL,
    genreId     INT          NOT NULL,
    PRIMARY KEY (movieId, genreId),
    FOREIGN KEY (movieId)  REFERENCES Movies(movieId)  ON DELETE CASCADE,
    FOREIGN KEY (genreId)  REFERENCES Genres(genreId)  ON DELETE CASCADE
);

-- ------------------------------------------------------------
-- 5. Ratings
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Ratings (
    ratingId    INT          NOT NULL AUTO_INCREMENT,
    userId      INT          NOT NULL,
    movieId     INT          NOT NULL,
    rating      FLOAT        NOT NULL CHECK (rating BETWEEN 0.5 AND 5.0),
    ratedAt     DATETIME     NOT NULL,
    PRIMARY KEY (ratingId),
    UNIQUE KEY uq_ratings_user_movie (userId, movieId),
    FOREIGN KEY (userId)   REFERENCES Users(userId)    ON DELETE CASCADE,
    FOREIGN KEY (movieId)  REFERENCES Movies(movieId)  ON DELETE CASCADE,
    INDEX idx_ratings_userId  (userId),
    INDEX idx_ratings_movieId (movieId)
);

-- ------------------------------------------------------------
-- 6. Tags
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Tags (
    tagId       INT          NOT NULL AUTO_INCREMENT,
    userId      INT          NOT NULL,
    movieId     INT          NOT NULL,
    tag         VARCHAR(500) NOT NULL,
    taggedAt    DATETIME     NOT NULL,
    PRIMARY KEY (tagId),
    UNIQUE KEY uq_tags_user_movie_tag (userId, movieId, tag),
    FOREIGN KEY (userId)   REFERENCES Users(userId)    ON DELETE CASCADE,
    FOREIGN KEY (movieId)  REFERENCES Movies(movieId)  ON DELETE CASCADE,
    INDEX idx_tags_movieId (movieId)
);

-- ------------------------------------------------------------
-- 7. MovieLinks
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS MovieLinks (
    movieId     INT          NOT NULL,
    imdbId      VARCHAR(20),
    tmdbId      VARCHAR(20),
    PRIMARY KEY (movieId),
    FOREIGN KEY (movieId)  REFERENCES Movies(movieId)  ON DELETE CASCADE
);

-- ------------------------------------------------------------
-- 8. Recommendations  (lưu lịch sử gợi ý - UC08)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Recommendations (
    recId           INT          NOT NULL AUTO_INCREMENT,
    userId          INT          NOT NULL,
    movieId         INT          NOT NULL,
    method          ENUM('content_based', 'collaborative', 'hybrid') NOT NULL,
    score           FLOAT,
    recommendedAt   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (recId),
    FOREIGN KEY (userId)   REFERENCES Users(userId)    ON DELETE CASCADE,
    FOREIGN KEY (movieId)  REFERENCES Movies(movieId)  ON DELETE CASCADE,
    INDEX idx_rec_userId (userId)
);
