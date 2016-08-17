(ns coursera-ml.linear-regression.assignment6
  (:require [incanter.core :as incanter]
            [clojure.core.matrix :as mat]
            [coursera-ml.linear-regression.utils :as utils]
            [coursera-ml.linear-regression.reader :as house-data]))


(def features ["bedrooms"
               "bathrooms"
               "sqft_living"
               "sqft_lot"
               "floors"
               "waterfront"
               "view"
               "condition"
               "grade"
               "sqft_above"
               "sqft_basement"
               "yr_built"
               "yr_renovated"
               "lat"
               "long"
               "sqft_living15"
               "sqft_lot15"])

(def train-data (utils/get-data house-data/small-train-data features :price))
(def norms (utils/feature-norms (get train-data 0)))
(def features-train (utils/normalize-features (get train-data 0) norms))
(def test-data (utils/get-data house-data/small-test-data features :price))
(def features-test (utils/normalize-features (get test-data 0) norms))
(def validation-data (utils/get-data house-data/small-validation-data features :price))
(def features-valid (utils/normalize-features (get validation-data 0) norms))

(defn euclid-distance
  [a b]
  (->> (mat/sub a b)
       mat/square
       incanter/sum
       Math/sqrt))

(defn compute-distances
  [train-features target-features]
  (map-indexed (fn [i train]
                 [i (euclid-distance train target-features)])
               train-features))

(defn k-nearest-neighbors
  [k features-train features-query]
  (->> (compute-distances features-train features-query)
       (sort-by second)
       (take k)
       (map first)))

(defn predict-output
  [k features-train output-train features-query]
  (for [feature-query features-query]
    (/ (->> (k-nearest-neighbors k features-train feature-query)
            (map #(nth output-train %))
            (apply +)) k)))

#_(euclid-distance (get features-train 9) (get features-test 0))
#_(for [i (range 0 10)]
    (euclid-distance (get features-train i) (get features-test 0)))

#_(for [i (range 0 10)]
    (euclid-distance (get features-train i) (get features-test 2)))

#_(def diff (map #(mat/sub % (get features-test 0)) features-train))
#_(incanter/sum (first (reverse diff)))

#_(apply (partial min-key second) (compute-distances features-train (get features-test 2)))
#_(-> (second train-data) (get 382))

#_(k-nearest-neighbors 4 features-train (features-test 2))
#_ (predict-output 4 features-train (get train-data 1) [(get features-test 2)])
#_(predict-output 10 features-train (get train-data 1) (take 10 features-test))
#_(def predictions (pmap
                    #(predict-output % features-train (get train-data 1) features-valid)
                    (range 1 16)))
#_(def rss-predictions (pmap #(utils/rss % (get validation-data 1)) predictions))
#_(apply min rss-predictions)



