(ns coursera-ml.linear-regression.assignment-1
  (:require [incanter.core :as incanter]
            [incanter.io :as io]))

(def train-data (io/read-dataset "/media/sf_rohit-mount/mine/coursera-ml/data/kc_house_train_data.csv"
                                 :header true))

(defn get-column [ds column]
  (incanter/$ column ds))

(def price-y (get-column train-data :price))
(def sqft-x (get-column train-data :sqft_living))
(def bedrooms-x (get-column train-data :bedrooms))

(defn closed-form-linear-regression
  [col-x col-y]
  (let [sum-x (apply + col-x)
        sum-y (apply + col-y)
        n (count col-x)
        sum-product-xy (apply + (map * col-x col-y))
        sum-x-2 (apply + (map #(Math/pow % 2) col-x))
        numerator (- sum-product-xy (/ (* sum-x sum-y) n))
        denominator (- sum-x-2 (/ (* sum-x sum-x) n))]
    [(/ numerator denominator) (- (/ sum-y n) (* (/ numerator denominator) (/ sum-x n)))]))

(defn predict-price
  [slope intercept x-val]
  (+ intercept (* x-val slope)))

(defn rss
  [slope intercept col-x col-y]
  (->> (map #(predict-price slope intercept %) col-x)
       (map (fn [actual predicted] (Math/pow (- actual predicted) 2)) col-y)
       (apply +)))

#_(let [[slope intercept] (closed-form-linear-regression sqft-x price-y)
        [s1 i1] (closed-form-linear-regression bedrooms-x price-y)]
    (println slope intercept)
    (println (predict-price slope intercept 2650))
    (println (rss slope intercept sqft-x price-y))
    (println (/ (- 800000 intercept) slope))
    (println (rss s1 i1 bedrooms-x price-y)))

