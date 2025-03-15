(ns cyclone-track-forecasting.core
  (:require [scicloj.kindly.v4.kind :as kind]
            [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.metamorph :as ds-mm]
            [scicloj.metamorph.core :as mm]
            [scicloj.metamorph.ml :as ml]
            [scicloj.metamorph.ml.loss :as loss]
            [scicloj.ml.smile.regression]))

; ### Load and process Dataset

(def ds (ds/->dataset "resources/final_dataset.csv" {:key-fn keyword}))

(def ds-intermediate (tc/drop-columns ds [:name :basin :filename :timestamp]))
(def ds-final-dx (tc/drop-missing (tc/drop-columns ds-intermediate [:dy])))
(def ds-final-dy (tc/drop-missing (tc/drop-columns ds-intermediate [:dx])))

(defn train-loop 
  [ds-intermediate model-name]
)

(defn predict-loop
  [model-dx model-dy])


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