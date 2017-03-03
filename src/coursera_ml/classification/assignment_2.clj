(ns coursera-ml.classification.assignment-2
  (:require [incanter.io :as io]
            [clojure.string :as str]
            [cheshire.core :as json]
            [clojure.core.matrix :as mat]
            [nd4clj.matrix :as matimp]
            [incanter.core :as incanter]))


(mat/set-current-implementation :nd4j)

(defn remove-punctuation
  [word]
  (str/replace word #"(?i)[^\w]+" ""))

(defn clean-dataset
  [data-set]
  (for [review data-set]
    (->> (map remove-punctuation (str/split (:review review) #" "))
         (str/join " ")
         (assoc review :review_clean))))

(defn add-na
  [data-set]
  (for [review data-set]
    (if (empty? (:review review))
      (assoc review :review "NA")
      review)))

(defn ignore-neutral-sentimant
  [data-set]
  (filter #(not= 3 (:rating %)) data-set))

(defn assign-sentiment
  [data-set]
  (for [review data-set]
    (if (> (:rating review) 3)
      (assoc review :sentiment 1)
      (assoc review :sentiment -1))))

(defn create-split
  [data-set indices]
  (map (partial nth data-set) indices))

(defn add-important-word-count
  [important-words prepared-data]
  (for [review prepared-data]
    (as-> (str/split (:review_clean review) #" ") $
          (group-by identity $)
          (map (fn [[word vals]] [word (count vals)]) $)
          (into {} $)
          (select-keys $ important-words)
          (assoc review :word-count $))))

(defn make-features-matrix
  [prepared-data important-words]
  (let [empty-row (apply vector (repeat (count important-words) 0))]
    (for [{:keys [word-count]} prepared-data]
      (reduce (fn [acc [word count]] (assoc acc (.indexOf important-words word) count)) empty-row word-count))))

(defn prepare-data
  [data-set important-words]
  (->> data-set
       add-na
       clean-dataset
       (add-important-word-count important-words)))

(defn features-matrix
  [prepared-data important-words]
  (let [m (make-features-matrix prepared-data important-words)
        sentiment (map :sentiment prepared-data)
        all-ones (repeat (count prepared-data) 1)]
    [(mat/matrix (mat/transpose (cons all-ones (mat/transpose m)))) (vec sentiment)]))

(defn prediction
  [score]
  (/ 1 (+' 1 (Math/pow Math/E (*' -1 score)))))

(defn predict-probability
  [feature-matrix coefficients]
  (as-> (mat/mmul feature-matrix (mat/transpose coefficients)) $
       (mat/get-column $ 0)
       (map prediction $)))

(defn feature-derivative
  [feature errors]
  (->> (mat/column-matrix errors)
       (mat/mmul feature)))

(defn log-likelihood
  [feature-matrix sentiment coefficients]
  (let [scores (mat/get-column (mat/mmul feature-matrix (mat/transpose coefficients)) 0)
        log-exp (map #(let [l (Math/log (inc (Math/pow Math/E (* -1 %))))] (if (Double/isNaN l) (* -1 %) l)) scores)
        indicators (vec (map #(if (> % 0) 0 -1) sentiment))
        first-term (incanter/to-vect (mat/transpose (mat/mul scores indicators)))]
    (apply + (mat/sub first-term log-exp))))


(defn prediction-errors
  [prediction-probability sentiment]
  (letfn [(error [s p]
            (if (> s 0) (- 1 p) (* -1 p)))]
    (map-indexed (fn [i p] (error (nth sentiment i) p)) prediction-probability)))

(defn logistic-regression
  [[feature-matrix sentiment] coeff-matrix step-size max-iter]
  (if (zero? max-iter)
    coeff-matrix
    (let [predictions (predict-probability feature-matrix coeff-matrix)
          errors (prediction-errors predictions sentiment)
          coefficients (mat/transpose (vec (for [j (range (mat/column-count feature-matrix))
                                              :let [fd (feature-derivative  (mat/get-column feature-matrix j) errors)]]
                                          (mat/scale step-size fd))))]
      (when (zero? (mod max-iter 30))
        (println (log-likelihood feature-matrix sentiment coefficients)))
      (recur [feature-matrix sentiment] coefficients step-size (dec max-iter)))))

(def data-set
  (:rows (io/read-dataset "/media/sf_rohit-mount/mine/coursera-ml/data/amazon_baby_subset.csv" :header true)))

(def test-indexes
  (json/parse-stream (clojure.java.io/reader "/media/sf_rohit-mount/mine/coursera-ml/data/module-2-assignment-test-idx.json") true))

(def train-indexes
  (json/parse-stream (clojure.java.io/reader "/media/sf_rohit-mount/mine/coursera-ml/data/module-2-assignment-train-idx.json") true))

(def important-words
  (vec
    (json/parse-stream (clojure.java.io/reader "/media/sf_rohit-mount/mine/coursera-ml/data/important_words.json") true)))

(def prepared-data
  (prepare-data data-set important-words))

(def matrix-data (features-matrix prepared-data important-words))
;;number of reviews containing perfect
#_(count (filter
           (fn [{:keys [word-count]}] (contains? word-count "perfect"))
           prepared-data))

;;number of features
#_(count important-words)

;;do logisitic regression
#_ (def coefficients
     (logistic-regression
       (features-matrix prepared-data important-words)
       (mat/matrix (vec (vec (repeat (inc (count important-words)) 0))))
       (Math/pow 10 -7)
       301))
(def a (mat/matrix [[1 2 3] [1 -1 -1]]))
(def s [-1 1])
(def c (mat/matrix [[1 3 -1]]))
#_(logistic-regression [a s] c 1 10)
