(ns cyclone-track-forecasting.core
  (:require [scicloj.kindly.v4.kind :as kind]
            [tablecloth.api :as tc])
  )

; # This is a test

(defn load-dataset 
  "Loads the dataset"
  [path]
  (tc/dataset path)
  )
