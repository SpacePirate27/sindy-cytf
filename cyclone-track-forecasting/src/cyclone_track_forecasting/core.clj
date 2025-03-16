; ### Imports

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
  (let [ds-dx-drop (tc/drop-columns ds-intermediate [:dy])
        ds-dy-drop (tc/drop-columns ds-intermediate [:dx])
        ds-dx (tc/drop-missing ds-dx-drop)
        ds-dy (tc/drop-missing ds-dy-drop)
        split-x (first (tc/split->seq ds-dx :holdout {:seed 112723}))
        split-y (first (tc/split->seq ds-dy :holdout {:seed 112723}))
        prediction-x (-> (:test split-x) (mm/transform-pipe pipeline-x model-dx) :metamorph/data :dx)
        prediction-y (-> (:test split-y) (mm/transform-pipe pipeline-y model-dy) :metamorph/data :dy)] 
  [prediction-x prediction-y]))


(def models [:smile.regression/ordinary-least-square :smile.regression/gradient-tree-boost :smile.regression/random-forest])

(defn mean-absolute-error 
  [y-true y-pred]
  (stats/mean (map #(Math/abs (- %1 %2)) y-true y-pred)))

; ### Training Models
(defn evaluate-models 
  [ds-intermediate] 
  (doseq [model models]
    (let [[fitted-x fitted-y pipeline-x pipeline-y] (train-loop ds-intermediate model)
          [pred-x pred-y] (predict-loop fitted-x fitted-y pipeline-x pipeline-y ds-intermediate)
          true-x (tc/column (:test (first (tc/split->seq (tc/drop-missing (tc/drop-columns ds-intermediate [:dy])) :holdout {:seed 112723}))) :dx)
          true-y (tc/column (:test (first (tc/split->seq (tc/drop-missing (tc/drop-columns ds-intermediate [:dx])) :holdout {:seed 112723}))) :dy)
          mae-x (mean-absolute-error true-x pred-x)
          mae-y (mean-absolute-error true-y pred-y)]
      {:model model :mae-x mae-x :mae-y mae-y})))


(evaluate-models ds-intermediate)

(kind/reagent
 ['(fn []
     [:div {:style {:height "500px"}
            :ref   (fn [el]
                     (let [m (-> js/L
                                 (.map el)
                                 (.setView (clj->js [12.97 77.59])
                                           13))
                           lat-longs [[15.0 68.0]
                                      [15.0 67.0]
                                      [15.0 66.5]
                                      [18.0 66.0]
                                      [20.0 64.0]]
                           polyline (-> js/L
                                        (.polyline (clj->js lat-longs) (clj->js {:color "red"}))
                                        (.addTo m))]
                       (-> js/L
                           .-tileLayer
                           (.provider "OpenStreetMap.Mapnik")
                           (.addTo m))
                       (.fitBounds m (.getBounds polyline))))}])]
 {:html/deps [:leaflet]})