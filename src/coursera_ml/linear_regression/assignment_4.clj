(ns coursera-ml.linear-regression.assignment-4
  (:require [clojure.core.matrix :as mat]
            [incanter.core :as incanter]
            [coursera-ml.linear-regression.reader :as house-data]
            [incanter.charts :as charts]))

(defn num-rows
  [data-frame]
  (-> (incanter/dim data-frame)
      first))

(defn get-data
  [data-frame input-features output]
  (let [all-ones (repeat (num-rows data-frame) 1)
        features-matrix (mat/transpose (cons all-ones (map #(incanter/$ % data-frame) input-features)))
        output-vector (vec (incanter/$ output data-frame))]
    [features-matrix output-vector]))

(defn predictions
  [features-matrix weights]
  (mat/mul features-matrix (mat/transpose weights)))


(defn feature-derivative-ridge
  [feature errors weight l2-penalty constant?]
  (cond-> (incanter/mult 2 (mat/mul feature errors))
          constant? (identity)
          (not constant?) (+' (*' 2 l2-penalty weight))))

(defn feature-derivatives
  [features-matrix errors weights l2-penalty]
  (for [i (range 0 (count (features-matrix 0)))
        :let [feature (map #(% i) features-matrix)
              constant? (zero? i)]]
    (feature-derivative-ridge feature errors (get weights i) l2-penalty constant?)))

(defn ridge-regression-gradient-descent
  [[features-matrix output] weights step-size l2-penalty iterations]
  (if (zero? iterations)
    weights
    (let [run-predictions (predictions features-matrix weights)
          errors (mat/sub run-predictions output)
          derivatives (vec (feature-derivatives features-matrix errors weights l2-penalty))
          new-weights (vec (for [i (range 0 (count weights))] (-' (nth weights i) (*' step-size (get derivatives i)))))]
      (recur [features-matrix output] new-weights step-size l2-penalty (dec iterations)))))

(defn rss [predictions output]
  (let [diff (mat/sub predictions output)
        transpose (mat/transpose diff)]
    (mat/mul diff transpose)))

(def data (get-data house-data/train-data [:sqft_living] :price))
(def test-data (get-data house-data/test-data [:sqft_living] :price))
(def data-model1 (get-data house-data/train-data [:sqft_living :sqft_living15] :price))
(def test-data-model1 (get-data house-data/test-data [:sqft_living :sqft_living15] :price))
#_(def simple-weights-0-penalty (ridge-regression-gradient-descent
                                data
                               [0 0]
                               (Math/pow 10 -12)
                               0
                               1000))

#_(def simple-weights-high-penalty (ridge-regression-gradient-descent
                                data
                                [0 0]
                                (Math/pow 10 -12)
                                (Math/pow 10 11)
                                1000))

#_(def plot (charts/scatter-plot (incanter/$ :sqft_living house-data/train-data)
                               (incanter/$ :price house-data/train-data)))

#_(incanter/view plot)
#_(charts/add-lines plot (incanter/$ :sqft_living house-data/train-data) (predictions (get data 0) simple-weights-0-penalty))
#_(charts/add-lines plot (incanter/$ :sqft_living house-data/train-data) (predictions (get data 0) simple-weights-high-penalty))
#_(rss (predictions (get test-data 0) [0 0])
     (get test-data 1))
#_(rss (predictions (get test-data 0) simple-weights-0-penalty)
        (get test-data 1))
#_(rss (predictions (get test-data 0) simple-weights-high-penalty)
     (get test-data 1))
#_(def multiple-weights-0-penalty (ridge-regression-gradient-descent
                                  data-model1
                                  [0 0 0]
                                  (Math/pow 10 -12)
                                  0
                                  1000))
#_(def multiple-weights-high-penalty (ridge-regression-gradient-descent
                                   data-model1
                                   [0 0 0]
                                   (Math/pow 10 -12)
                                   (Math/pow 10 11)
                                   1000))
#_(rss (predictions (get test-data-model1 0) [0 0 0])
     (get test-data 1))
#_(rss (predictions (get test-data-model1 0) multiple-weights-0-penalty)
        (get test-data 1))
#_(rss (predictions (get test-data-model1 0) multiple-weights-high-penalty)
     (get test-data 1))

#_(get (predictions (get test-data-model1 0) multiple-weights-0-penalty) 0)
#_(get (predictions (get test-data-model1 0) multiple-weights-high-penalty) 0)
