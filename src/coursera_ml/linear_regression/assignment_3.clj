(ns coursera-ml.linear-regression.assignment-3
  (:require [incanter.core :as incanter]
            [incanter.stats :as stats]
            [incanter.charts :as charts]
            [coursera-ml.linear-regression.reader :as house-data]
            [clojure.core.matrix :as mat]))

(defn x-axis
  [df]
  (incanter/$ :sqft_living df))

(defn pow-feature
  [x-column degree]
  (if (<= degree 1)
    x-column
    (reduce incanter/bind-columns
            (for [i (range 1 (inc degree))]
              (incanter/pow x-column i)))))

(defn do-linear-regression
  [output features]
  (stats/linear-model output features))

(defn model-degree
  [degree df]
  (do-linear-regression
    (incanter/$ :price df)
    (pow-feature (incanter/$ :sqft_living df) degree)))

#_(def model-1-degree (model-degree 1 house-data/all-data))
#_(def model-2-degree (model-degree 2 house-data/all-data))
#_(def model-3-degree (model-degree 3 house-data/all-data))
#_(def model-15-degree (model-degree 15 house-data/all-data))

(def model-15-set1 (model-degree 15 house-data/wk3-set1))
(def model-15-set2 (model-degree 15 house-data/wk3-set2))
(def model-15-set3 (model-degree 15 house-data/wk3-set3))
(def model-15-set4 (model-degree 15 house-data/wk3-set4))

(defn draw-plot
  [source-data model]
  (let [plot (charts/scatter-plot (incanter/$ :sqft_living source-data)
                                  (incanter/$ :price source-data))]
    (incanter/view plot)
    (charts/add-lines plot (x-axis source-data) (:fitted model))))

(defn predictions
  [model raw-values]
  (map #(stats/predict model %) raw-values))



#_(def plot1 (charts/scatter-plot (incanter/$ :sqft_living house-data/all-data)
                                  (incanter/$ :price house-data/all-data)))
#_(incanter/view plot1)
#_(charts/add-lines plot1 (x-axis house-data/all-data) (:fitted model-1-degree))
#_(charts/add-lines plot1 (x-axis house-data/all-data) (:fitted model-2-degree))
#_(charts/add-lines plot1 (x-axis house-data/all-data) (:fitted model-3-degree))
#_(charts/add-lines plot1 (x-axis house-data/all-data) (:fitted model-15-degree))

#_(:coefs model-15-set1)
#_(:coefs model-15-set2)
#_(:coefs model-15-set3)
#_(:coefs model-15-set4)

#_(draw-plot house-data/wk3-set1 model-15-set1)
#_(draw-plot house-data/wk3-set2 model-15-set2)
#_(draw-plot house-data/wk3-set3 model-15-set3)
#_(draw-plot house-data/wk3-set4 model-15-set4)

(def validation-sets-with-model
  (for [i (range 0 15)]
    [(pow-feature (x-axis house-data/wk3-valid-data) i)
     (do-linear-regression (incanter/$ :price house-data/wk3-train-data)
                           (pow-feature (x-axis house-data/wk3-train-data) i))]))

(defn rss
  [target-prices predictions]
  (apply + (incanter/pow (mat/sub target-prices predictions) 2)))

(def best-model
  (->> (map (fn [[raw-matrix model]]
              [(rss (incanter/$ :price house-data/wk3-valid-data)
                    (predictions model raw-matrix))
               model]) validation-sets-with-model)
       (apply min-key first)))

(def rss-for-best-model
  (let [degree (-> (second best-model) :coefs count dec)
        target-price (incanter/$ :price house-data/wk3-test-data)]
    (rss target-price
         (predictions (second best-model) (pow-feature (x-axis house-data/wk3-test-data) degree)))))