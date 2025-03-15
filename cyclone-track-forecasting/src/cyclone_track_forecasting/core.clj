(ns cyclone-track-forecasting.core
  (:require [scicloj.kindly.v4.kind :as kind]
            [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.metamorph :as ds-mm]
            [scicloj.metamorph.core :as mm]
            [scicloj.metamorph.ml :as ml]
            [scicloj.metamorph.ml.loss :as loss]
            [scicloj.ml.smile.regression])
  )



; ### Load and process Dataset

(def ds (ds/->dataset "resources/final_dataset.csv" {:key-fn keyword}))

(def ds-intermediate (tc/drop-columns ds [:name :basin :filename :timestamp]))
(def ds-final (tc/drop-missing ds-intermediate))


; ### Train-Test split
(def split
  (first
   (tc/split->seq ds-final :holdout {:seed 112723})))

(:train split)
(:test split)

; ### Pipeline
(def ols-pipe-fn
  (mm/pipeline
   (ds-mm/set-inference-target :dx)
   #:metamorph{:id :model}
   (ml/model {:model-type :smile.regression/ordinary-least-square})))


; ### Fit Regressor
(def fitted (mm/fit (:train split)  ols-pipe-fn))

; ### Get Prediction
(def prediction
  (-> (:test split)
      (mm/transform-pipe ols-pipe-fn fitted)
      :metamorph/data
      :dx))

prediction