(ns coursera-ml.linear-regression.utils
  (:require [incanter.core :as incanter]
            [clojure.core.matrix :as mat]))

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

(defn feature-norms
  [feature-matrix]
  (->> (mat/transpose feature-matrix)
       (map #(map (fn [x] (* x x)) %))
       (map #(apply + %))
       (map #(Math/sqrt %))
       vec))

(defn normalize-features
  [feature-matrix norms]
  (->> (mat/transpose feature-matrix)
       (map-indexed (fn [i col] (map #(/ % (get norms i)) col)))
       (mat/transpose)))

(defn rss [predictions output]
  (let [diff (mat/sub predictions output)
      transpose (mat/transpose diff) ]
    (mat/mul diff transpose)))

