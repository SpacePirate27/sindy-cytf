(ns sindy-cytf.leaflet
  (:require [scicloj.kindly.v4.kind :as kind]))

(defn plot-map
  [org-lat-lons pred-lat-lons]
  (kind/reagent
    ['(fn [lat-longs lat-longs-pred]
        [:div {:style {:height "500px"}
               :ref   (fn [el]
                        (let [m (-> js/L
                                    (.map el)
                                    (.setView (clj->js [12.97 77.59])
                                              13))
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
                          (.fitBounds m (.getBounds polyline-pred))))}])
     org-lat-lons pred-lat-lons]
    {:html/deps [:leaflet]}))

(plot-map [[10.0 90.5] [10.8 90.5] [11.7 90.5] [11.7 90.5]
           [11.8 90.5] [12.0 90.5] [12.2 90.0] [12.2 89.9]
           [12.3 89.6] [12.5 89.0] [13.2 86.4] [13.3 85.3]
           [13.3 85.0] [13.3 83.0] [13.3 82.5] [13.1 82.3]
           [13.2 81.9] [13.0 79.9] [12.9 79.5] [12.7 79.1]]
          [[10.51733251 90.00463473] [11.13929305 89.81331401] [12.38654183 90.20973648] [12.28330113 90.41288601]
           [12.12821909 90.05590464] [12.42509118 90.47215197] [12.68738628 89.66484756] [12.47824889 90.03896979]
           [12.56855659 89.35409665] [12.94705222 88.62794991] [13.80642426 85.59069015] [13.37532648 84.81350484]
           [13.43248887 84.45644182] [13.67090765 82.43335564] [13.71508199 81.80330209] [13.36208945 81.72841881]
           [13.35540318 81.43748273] [13.50931791 79.02517984] [12.70298589 79.62044354] [12.82211828 79.35065624]])

