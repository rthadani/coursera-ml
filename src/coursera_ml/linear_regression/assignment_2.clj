(ns coursera-ml.linear-regression.assignment-2
  (:require [clojure.core.matrix :as mat]
            [incanter.core :as incanter]
            [incanter.io :as io]
            [coursera-ml.linear-regression.utils :as utils]
            [coursera-ml.linear-regression.reader :as house-data]))


(defn transpose
  [v]
  (apply mapv vector v))

(defn predictions
  [features-matrix weights]
  (mat/mul features-matrix (mat/transpose weights)))

(defn feature-derivative
  [feature errors]
  (incanter/mult 2 (mat/mul feature errors)))

(defn feature-derivatives
  [features-matrix errors]
  (for [i (range 0 (count (features-matrix 0)))
        :let [feature (map #(% i) features-matrix)]]
    (feature-derivative feature errors)))

(defn converged? [gradient-magnitude tolerance]
  (< gradient-magnitude tolerance))

(defn regression-gradient-descent
  [features-matrix output weights step-size tolerance gradient-magnitude]
  (println weights gradient-magnitude)
  (if (converged? gradient-magnitude tolerance)
    weights
    (let [run-predictions (predictions features-matrix weights)
          errors (mat/sub run-predictions output)
          derivatives (vec (feature-derivatives features-matrix errors))
          gradient-sum-square (reduce (fn [acc derivative] (+ acc (Math/pow derivative 2))) 0 derivatives)
          new-weights (map-indexed (fn [i w] (- w (* step-size (derivatives i)))) weights)]
      (recur features-matrix output new-weights step-size tolerance (Math/sqrt gradient-sum-square)) )))


#_(utils/get-data house-data/train-data [:sqft_living] :price)
#_(def features-matrix (*1 0))
#_(def output (*2 1))
#_(def initial-weights [-47000 1])
#_(def step-size (*' 7 (Math/pow 10 -12)))
#_(def tolerance (*' 2.5 (Math/pow 10 7)))
#_(regression-gradient-descent features-matrix output initial-weights step-size tolerance (Double/MAX_VALUE))
#_(def output-weights *1)
#_(utils/get-data house-data/test-data [:sqft_living] :price)
#_(def test-features (*1 0))
#_(def test-price (*2 1))
#_ (predictions (get test-features 0) output-weights)
#_ (utils/rss (predictions test-features output-weights) test-price)
;=> 2.7540004490212878E14

#_(utils/get-data house-data/train-data [:sqft_living :sqft_living15] :price)
#_(def features-matrix (*1 0))
#_(def output (*2 1))
#_(def initial-weights [-100000 1 1])
#_(def step-size (*' 4 (Math/pow 10 -12)))
#_(def tolerance (*' 1 (Math/pow 10 9)))
#_(regression-gradient-descent features-matrix output initial-weights step-size tolerance (Double/MAX_VALUE))
#_(def output-weights *1)
#_(utils/get-data house-data/test-data [:sqft_living :sqft_living15] :price)
#_(def test-features (*1 0))
#_(def test-price (*2 1))
#_ (predictions (get test-features 0) output-weights)
#_ (utils/rss (predictions test-features output-weights) test-price)


