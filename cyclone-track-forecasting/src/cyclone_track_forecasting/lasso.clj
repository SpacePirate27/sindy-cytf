(ns cyclone-track-forecasting.lasso
  (:require [scicloj.kindly.v4.kind :as kind]
            [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.metamorph :as ds-mm]
            [scicloj.metamorph.core :as mm]
            [scicloj.metamorph.ml :as ml]
            [scicloj.ml.smile.regression]
            [fastmath.stats :as stats]))

(def ds (ds/->dataset "resources/final_dataset.csv" {:key-fn keyword}))

(def ds-intermediate (tc/drop-columns ds [:name :basin :filename :timestamp]))
(def dx-intermediate (tc/drop-missing (tc/drop-columns ds-intermediate [:dy])))
(def dy-intermediate (tc/drop-missing (tc/drop-columns ds-intermediate [:dx])))

; ## Lasso - dx
(defn make-pipe-fn-dx [lambda]
  (mm/pipeline 
   (ds-mm/set-inference-target :dx)
   #:metamorph{:id :model}
   (ml/model
    {:model-type :smile.regression/lasso, :lambda (double lambda)})))


(def coefs-vs-lambda-dx
  (flatten
   (map
    (fn [lambda]
      (let [fitted (mm/fit-pipe dx-intermediate (make-pipe-fn-dx lambda))
            model-instance (-> fitted :model (ml/thaw-model))
            predictors (map
                        #(first (.variables %))
                        (seq
                         (.. model-instance formula predictors)))]
        (map
         #(hash-map
           :log-lambda
           (Math/log10 lambda)
           :coefficient
           %1
           :predictor
           %2)
         (-> model-instance .coefficients seq)
         predictors)))
    (range 1 100000 100))))

(kind/vega-lite
 {:data {:values coefs-vs-lambda-dx},
  :width 500,
  :height 500,
  :mark {:type "line"},
  :encoding
  {:x {:field :log-lambda, :type "quantitative"},
   :y {:field :coefficient, :type "quantitative"},
   :color {:field :predictor}}})


; ## Lasso - dy
(defn make-pipe-fn-dy [lambda]
  (mm/pipeline 
   (ds-mm/set-inference-target :dy)
   #:metamorph{:id :model}
   (ml/model
    {:model-type :smile.regression/lasso, :lambda (double lambda)})))

(def coefs-vs-lambda-dy
  (flatten
   (map
    (fn [lambda]
      (let [fitted (mm/fit-pipe dy-intermediate (make-pipe-fn-dy lambda))
            model-instance (-> fitted :model (ml/thaw-model))
            predictors (map
                        #(first (.variables %))
                        (seq
                         (.. model-instance formula predictors)))]
        (map
         #(hash-map
           :log-lambda
           (Math/log10 lambda)
           :coefficient
           %1
           :predictor
           %2)
         (-> model-instance .coefficients seq)
         predictors)))
    (range 1 100000 100))))

(kind/vega-lite
 {:data {:values coefs-vs-lambda-dy},
  :width 500,
  :height 500,
  :mark {:type "line"},
  :encoding
  {:x {:field :log-lambda, :type "quantitative"},
   :y {:field :coefficient, :type "quantitative"},
   :color {:field :predictor}}})