(ns cyclone-track-forecasting.core
  (:require [scicloj.kindly.v4.kind :as kind]
            [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.metamorph :as ds-mm]
            [scicloj.metamorph.core :as mm]
            [scicloj.metamorph.ml :as ml]
            [scicloj.ml.smile.regression]
            [fastmath.stats :as stats]))

; ### Load Dataset

(def ds (ds/->dataset "resources/final_dataset.csv" {:key-fn keyword}))

(def ds-intermediate (tc/drop-columns ds [:name :basin :filename :timestamp]))

(def dx-intermediate (tc/drop-missing (tc/drop-columns ds-intermediate [:dy])))
(def dy-intermediate (tc/drop-missing (tc/drop-columns ds-intermediate [:dx])))

; ### Training the Model
(defn train-loop 
  [ds-intermediate model-name]
  (let [ds-dx (tc/drop-missing (tc/drop-columns ds-intermediate [:dy]))
        ds-dy (tc/drop-missing (tc/drop-columns ds-intermediate [:dx]))
        split-x (first (tc/split->seq ds-dx :holdout {:seed 112723 :ratio 0.8}))
        split-y (first (tc/split->seq ds-dy :holdout {:seed 112723 :ratio 0.8}))
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
    [fitted-x fitted-y pipeline-x pipeline-y]))

(defn predict-loop
  [model-dx model-dy pipeline-x pipeline-y ds-intermediate]
  (let [ds-dx (tc/drop-missing (tc/drop-columns ds-intermediate [:dy]))
        ds-dy (tc/drop-missing (tc/drop-columns ds-intermediate [:dx]))
        split-x (first (tc/split->seq ds-dx {:seed 112723}))
        split-y (first (tc/split->seq ds-dy {:seed 112723}))
        prediction-x (-> (:test split-x)(mm/transform-pipe pipeline-x model-dx) :metamorph/data :dx)
        prediction-y (-> (:test split-y)(mm/transform-pipe pipeline-y model-dy) :metamorph/data :dy)] 
  [prediction-x prediction-y]))


(def models [:smile.regression/ordinary-least-square])

(defn mean-absolute-error [y-true y-pred]
  (stats/mean (map #(Math/abs (- %1 %2)) y-true y-pred)))

; ### Training Models
(defn evaluate-models [ds-intermediate] 
  (doseq [model models]
    (let [[fitted-x fitted-y pipeline-x pipeline-y] (train-loop ds-intermediate model)
          [pred-x pred-y] (predict-loop fitted-x fitted-y pipeline-x pipeline-y ds-intermediate)
          true-x (tc/column (:test (first (tc/split->seq (tc/drop-missing (tc/drop-columns ds-intermediate [:dy])) {:seed 112723}))) :dx)
          true-y (tc/column (:test (first (tc/split->seq (tc/drop-missing (tc/drop-columns ds-intermediate [:dx])) {:seed 112723}))) :dy)
          mae-x (mean-absolute-error true-x pred-x)
          mae-y (mean-absolute-error true-y pred-y)]
      (println (format "Model: %s | MAE (dx): %.4f | MAE (dy): %.4f" model mae-x mae-y)))))

;; (evaluate-models ds-intermediate)


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