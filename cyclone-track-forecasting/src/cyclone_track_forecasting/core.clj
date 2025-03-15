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

(defn train-loop 
  [ds-intermediate model-name]
  (let [ds-dx (tc/drop-missing (tc/drop-columns ds-intermediate [:dy]))
        ds-dy (tc/drop-missing (tc/drop-columns ds-intermediate [:dx]))
        split-x (first (tc/split->seq ds-dx :holdout {:seed 112723}))
        split-y (first (tc/split->seq ds-dy :holdout {:seed 112723}))
        pipeline-x (mm/pipeline
                    (ds-mm/set-inference-target :dx)
                    #:metamorph{:id :model}
                    (ml/model {:model-type model-name}))
        pipeline-y (mm/pipeline
                    (ds-mm/set-inference-target :dy)
                    #:metamorph{:id :model}
                    (ml/model {:model-type model-name}))
        fitted-x (mm/fit (:train split-x) pipeline-x)
        fitted-y (mm/fit (:train split-y) pipeline-y)]
    [fitted-x fitted-y]))

(defn predict-loop
  [model-dx model-dy])

; ### Get Prediction
(def prediction
  (-> (:test split)
      (mm/transform-pipe ols-pipe-fn fitted)
      :metamorph/data
      :dx))

prediction