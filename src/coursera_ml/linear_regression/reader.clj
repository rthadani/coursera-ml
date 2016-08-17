(ns coursera-ml.linear-regression.reader
  (:require [incanter.io :as io]))

(def base-path "/media/sf_rohit-mount/mine/coursera-ml/data")
(defn load-data
  [file]
  (io/read-dataset (str base-path "/" file) :header true))
(def train-data
  (load-data "kc_house_train_data.csv"))

(def test-data
  (load-data "kc_house_test_data.csv"))

(def all-data
  (load-data "kc_house_data.csv"))

(def wk3-set1
  (load-data "wk3_kc_house_set_1_data.csv"))
(def wk3-set2
  (load-data "wk3_kc_house_set_2_data.csv"))
(def wk3-set3
  (load-data "wk3_kc_house_set_3_data.csv"))
(def wk3-set4
  (load-data "wk3_kc_house_set_4_data.csv"))
(def wk3-train-data
  (load-data "wk3_kc_house_train_data.csv"))
(def wk3-test-data
  (load-data "wk3_kc_house_test_data.csv"))
(def wk3-valid-data
  (load-data "wk3_kc_house_valid_data.csv"))

#_(def assignment-3-train-data
  (load-data "wk3_kc_house_train_data"))
(def small-train-data
  (load-data "kc_house_data_small_train.csv"))

(def small-test-data
  (load-data "kc_house_data_small_test.csv"))

(def small-validation-data
  (load-data "kc_house_data_validation.csv"))

(def smallall-data
  (load-data "kc_house_data_small.csv"))
