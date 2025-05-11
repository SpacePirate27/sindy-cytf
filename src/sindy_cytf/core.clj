; # Tropical Cyclone Track Prediction
; ##### By [Naimish](https://github.com/Naimish240/) and [Ram Narayan](https://github.com/SpacePirate27/)

; ## 1. The Problem
; Tropical Cyclone's happen every year and claim hundreds of lives and billions in damages. Accurate cyclone track forecasting is crucial for disaster preparedness and mitigation. Timely predictions help authorities issue early warnings, evacuate affected areas, and minimize economic and human losses. This project focuses on forecasting cyclone tracks using historical data from the India Meteorological Department (IMD) and Wind Velocity Vectors calculated using EUMETSAT's METEOSAT-7 data.

; ## 2. Our Proposed Solution
; We attempt to solve this problem using Machine Learning and Deep Learning algorithms to forecast the trajectory of the cyclone using the following parameters:
; - Current Position of the Cyclone
; - Time of Observation (date, year and phase of moon)
; - Wind Velocities (in 0.5 degree bands around the centre of the cyclone)
; - Other Properties (Pressure, Strength, Surface Wind Speed, etc.)


; ## 3. Training the Models 
; ### Imports

(ns sindy-cytf.core
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
(def ds-intermediate (tc/drop-columns ds [:name :timestamp :lat :lon :filename]))

; ### Training Loop
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
        split-x (first (tc/split->seq ds-dx :holdout {:seed 112723 :ratio 0.8}))
        split-y (first (tc/split->seq ds-dy :holdout {:seed 112723 :ratio 0.8}))
        prediction-x (-> (:test split-x) (mm/transform-pipe pipeline-x model-dx) :metamorph/data :dx)
        prediction-y (-> (:test split-y) (mm/transform-pipe pipeline-y model-dy) :metamorph/data :dy)] 
  [prediction-x prediction-y]))

; We selected Ordinary Least Square, Gradient Tree Boost and Random Forest Regressors for training 
(def models [:smile.regression/ordinary-least-square :smile.regression/gradient-tree-boost :smile.regression/random-forest])

; And Mean absolute error as our metric
(defn mean-absolute-error 
  [y-true y-pred]
  (stats/mean (map #(Math/abs (- %1 %2)) y-true y-pred)))

; ### Evaluating Models
(defn evaluate-models
  [ds-intermediate]
  (mapv (fn [model]
          (let [[fitted-x fitted-y pipeline-x pipeline-y] (train-loop ds-intermediate model)
                [pred-x pred-y] (predict-loop fitted-x fitted-y pipeline-x pipeline-y ds-intermediate)
                true-x (tc/column (:test (first (tc/split->seq (tc/drop-missing (tc/drop-columns ds-intermediate [:dy])) :holdout {:seed 112723}))) :dx)
                true-y (tc/column (:test (first (tc/split->seq (tc/drop-missing (tc/drop-columns ds-intermediate [:dx])) :holdout {:seed 112723}))) :dy)
                mae-x (mean-absolute-error true-x pred-x)
                mae-y (mean-absolute-error true-y pred-y)
                _ (print "model : " model "\nmean absolute error (dx) : " mae-x "\nmean absolute error (dy) : " mae-y "\n---------------------------------\n")]
            {:model model :mae-x mae-x :mae-y mae-y :model-dx fitted-x :model-dy fitted-y :pipeline-dx pipeline-x :pipeline-dy pipeline-y}))
        models))

(def trained-models (evaluate-models ds-intermediate))


; ## 4. Predicting Trajectory for a Cyclone

(defn forecast-cyclone
  [model-dx model-dy pipeline-dx pipeline-dy cyclone-path]
  (let [ds (ds/->dataset cyclone-path {:key-fn keyword})
        ds-intermediate (tc/drop-columns ds [:name :basin :timestamp])
        ds-dx-drop (tc/dataset (tc/drop-missing (tc/drop-columns ds-intermediate [:dy])) {:result-type :as-seq})
        ds-dy-drop (tc/dataset (tc/drop-missing (tc/drop-columns ds-intermediate [:dx])) {:result-type :as-seq})
        split-x (first (tc/split->seq ds-dx-drop :holdout {:seed 112723 :ratio 0.99})) ; hack to load entire cyclone for prediction
        split-y (first (tc/split->seq ds-dy-drop :holdout {:seed 112723 :ratio 0.99}))
        prediction-dx (-> (:train split-x) (mm/transform-pipe pipeline-dx model-dx) :metamorph/data :dx)
        prediction-dy (-> (:train split-y) (mm/transform-pipe pipeline-dy model-dy) :metamorph/data :dy) 
        foo (tc/add-columns (:train split-x) {:dx-pred prediction-dx ; add lat to dx to get lat-pred
                                              :dy-pred prediction-dy}) ; add lon to dy to get lon-pred 
        bar (-> foo (tc/+ :lat-pred [:dx-pred :lat]))
        baz (-> bar (tc/+ :lon-pred [:dy-pred :lon]))]
    (kind/dataset (tc/select-columns baz [:filename :lat :lat-pred :lon :lon-pred]) {:dataset/print-range 100})))


; We take the Vardah cyclone as input to calculate the trajectory
(def outputs (mapv (fn [{:keys [model model-dx model-dy 
                   pipeline-dx pipeline-dy]}] 
        {model (forecast-cyclone 
         model-dx model-dy pipeline-dx pipeline-dy "resources/cyclones/VARDAH.csv")}) ;; change cyclone name here
      trained-models))

;; Due to limitations, we had to manually hard-code the positions after forecasting to plot them. We intend to work on a more elegant solution.


; ### Ordinary Least Square : Prediction
(kind/reagent
 ['(fn []
     [:div {:style {:height "500px"}
            :ref   (fn [el]
                     (let [m (-> js/L
                                 (.map el)
                                 (.setView (clj->js [12.97 77.59])
                                           13))
                           lat-longs [[10.0 90.5]
                                      [10.8 90.5]
                                      [11.7 90.5]
                                      [11.7 90.5]
                                      [11.8 90.5]
                                      [12.0 90.5]
                                      [12.2 90.0]
                                      [12.2 89.9]
                                      [12.3 89.6]
                                      [12.5 89.0]
                                      [13.2 86.4]
                                      [13.3 85.3]
                                      [13.3 85.0]
                                      [13.3 83.0]
                                      [13.3 82.5]
                                      [13.1 82.3]
                                      [13.2 81.9]
                                      [13.0 79.9]
                                      [12.9 79.5]
                                      [12.7 79.1]]
                           lat-longs-pred [[10.51733251 90.00463473]
                                           [11.13929305 89.81331401]
                                           [12.38654183 90.20973648]
                                           [12.28330113 90.41288601]
                                           [12.12821909 90.05590464]
                                           [12.42509118 90.47215197]
                                           [12.68738628 89.66484756]
                                           [12.47824889 90.03896979]
                                           [12.56855659 89.35409665]
                                           [12.94705222 88.62794991]
                                           [13.80642426 85.59069015]
                                           [13.37532648 84.81350484]
                                           [13.43248887 84.45644182]
                                           [13.67090765 82.43335564]
                                           [13.71508199 81.80330209]
                                           [13.36208945 81.72841881]
                                           [13.35540318 81.43748273]
                                           [13.50931791 79.02517984]
                                           [12.70298589 79.62044354]
                                           [12.82211828 79.35065624]]
                           polyline (-> js/L
                                        (.polyline (clj->js lat-longs) (clj->js {:color "red"}))
                                        (.addTo m))
                           polyline-pred (-> js/L
                                        (.polyline (clj->js lat-longs-pred) (clj->js {:color "green"}))
                                        (.addTo m))]
                       (-> js/L
                           .-tileLayer
                           (.provider "OpenStreetMap.Mapnik")
                           (.addTo m))
                       (.fitBounds m (.getBounds polyline))
                       (.fitBounds m (.getBounds polyline-pred))))}])]
 {:html/deps [:leaflet]})



; ### Gradient Tree Boost : Prediction
(kind/reagent
 ['(fn []
     [:div {:style {:height "500px"}
            :ref   (fn [el]
                     (let [m (-> js/L
                                 (.map el)
                                 (.setView (clj->js [12.97 77.59])
                                           13))
                           lat-longs [[10.0 90.5]
                                      [10.8 90.5]
                                      [11.7 90.5]
                                      [11.7 90.5]
                                      [11.8 90.5]
                                      [12.0 90.5]
                                      [12.2 90.0]
                                      [12.2 89.9]
                                      [12.3 89.6]
                                      [12.5 89.0]
                                      [13.2 86.4]
                                      [13.3 85.3]
                                      [13.3 85.0]
                                      [13.3 83.0]
                                      [13.3 82.5]
                                      [13.1 82.3]
                                      [13.2 81.9]
                                      [13.0 79.9]
                                      [12.9 79.5]
                                      [12.7 79.1]]
                           lat-longs-pred [[10.50441129 90.50081527]
                                           [11.24872929 90.50018621]
                                           [12.57168727 90.49372255]
                                           [11.71746603 90.50002262]
                                           [11.99223747 90.50010198]
                                           [12.21617285 90.49923645]
                                           [12.4386839 89.47279788]
                                           [12.21588498 89.8025816]
                                           [12.398839 89.30007349]
                                           [12.7297855 88.61510929]
                                           [13.92456363 85.32840367]
                                           [13.30196521 84.78979152]
                                           [13.27241693 84.69143272]
                                           [13.33336268 81.95916439]
                                           [13.19387995 82.1308043]
                                           [12.9266026 82.0112856]
                                           [13.12872709 81.50285386]
                                           [12.91652861 79.07708745]
                                           [12.71293184 79.13061112]
                                           [12.49887092 78.71074521]]
                           polyline (-> js/L
                                        (.polyline (clj->js lat-longs) (clj->js {:color "red"}))
                                        (.addTo m))
                           polyline-pred (-> js/L
                                             (.polyline (clj->js lat-longs-pred) (clj->js {:color "green"}))
                                             (.addTo m))]
                       (-> js/L
                           .-tileLayer
                           (.provider "OpenStreetMap.Mapnik")
                           (.addTo m))
                       (.fitBounds m (.getBounds polyline))
                       (.fitBounds m (.getBounds polyline-pred))))}])]
 {:html/deps [:leaflet]})


; ### Random Forest : Prediction
(kind/reagent
 ['(fn []
     [:div {:style {:height "500px"}
            :ref   (fn [el]
                     (let [m (-> js/L
                                 (.map el)
                                 (.setView (clj->js [12.97 77.59])
                                           13))
                           lat-longs [[10.0 90.5]
                                      [10.8 90.5]
                                      [11.7 90.5]
                                      [11.7 90.5]
                                      [11.8 90.5]
                                      [12.0 90.5]
                                      [12.2 90.0]
                                      [12.2 89.9]
                                      [12.3 89.6]
                                      [12.5 89.0]
                                      [13.2 86.4]
                                      [13.3 85.3]
                                      [13.3 85.0]
                                      [13.3 83.0]
                                      [13.3 82.5]
                                      [13.1 82.3]
                                      [13.2 81.9]
                                      [13.0 79.9]
                                      [12.9 79.5]
                                      [12.7 79.1]]
                           lat-longs-pred [[10.5920704 90.08543794]
                                           [11.28230111 90.37556548]
                                           [12.40336357 90.15933079]
                                           [11.85018825 90.36329667]
                                           [12.05135 90.37264683]
                                           [12.29353373 90.42026365]
                                           [12.69104397 89.29607421]
                                           [12.25024698 89.72631913]
                                           [12.55552532 89.38258952]
                                           [12.80443944 88.6309096]
                                           [13.89659238 84.86930556]
                                           [13.38830317 84.81995484]
                                           [13.33280778 84.5685846]
                                           [13.46567317 81.51696889]
                                           [13.29628968 82.12603706]
                                           [13.01000619 81.9624631]
                                           [13.18363254 81.41234587]
                                           [13.02896579 78.72375556]
                                           [12.89474714 79.14482325]
                                           [12.64996856 78.63623571]]
                           polyline (-> js/L
                                        (.polyline (clj->js lat-longs) (clj->js {:color "red"}))
                                        (.addTo m))
                           polyline-pred (-> js/L
                                             (.polyline (clj->js lat-longs-pred) (clj->js {:color "green"}))
                                             (.addTo m))]
                       (-> js/L
                           .-tileLayer
                           (.provider "OpenStreetMap.Mapnik")
                           (.addTo m))
                       (.fitBounds m (.getBounds polyline))
                       (.fitBounds m (.getBounds polyline-pred))))}])]
 {:html/deps [:leaflet]})

; ## 5. Challenges and Future work
; Long-term cyclone trajectory prediction remains a significant challenge due to the dynamic nature of atmospheric conditions. While regression models offer valuable insights, their predictive power can be enhanced by incorporating real-time data sources such as satellite imagery and advanced meteorological simulations. Another challenge is computational efficiency—deep learning models, while powerful, require substantial resources for training and deployment.
; Moving forward, this project aims to refine the predictive models by incorporating hybrid approaches that combine physics-based and data-driven techniques. The goal is to develop a scalable and interpretable forecasting system that can be integrated into disaster preparedness frameworks.
; One of the primary challenges in this study is handling the inherent uncertainty in cyclone paths, which requires models that can capture nonlinearity in the system. Future work will focus on training SINDy and deep learning models to address these gaps and enhance predictive capabilities.

; ## 6. Conclusion
; By establishing a strong baseline with regression models, we lay the groundwork for exploring advanced approaches such as SINDy and deep learning for Tropical Cyclone Track Forecasting. Enhanced predictive models will contribute to more effective early warning systems, ultimately reducing the impact of cyclones on vulnerable populations and infrastructure.