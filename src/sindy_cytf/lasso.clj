(ns sindy-cytf.lasso
  (:require [fastmath.stats :as stats]
            [scicloj.kindly.v4.kind :as kind]
            [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]
            [tech.v3.datatype :as dtype]
            [tech.v3.dataset.metamorph :as ds-mm]
            [scicloj.metamorph.core :as mm]
            [scicloj.metamorph.ml :as ml]
            [scicloj.ml.smile.regression]))


(def ds (ds/->dataset "resources/final_dataset.csv" {:key-fn keyword}))

(def ds-intermediate (tc/drop-columns ds [:name :basin :timestamp :lat :lon :filename :moon_phase :day_sine :year_sine]))

(defn product-augmented-map
  [m]
  (let [kvs (vec (seq m))
        n (count kvs)
        products (for [i (range n)
                       j (range i n)]
                   (let [[k1 v1] (nth kvs i)
                         [k2 v2] (nth kvs j)
                         new-key (keyword (str (name k1) "-" (name k2)))]
                     [new-key (* v1 v2)]))]
    (into {} products)))

(defn add-mult-combos-to-dataset
  [ds]
  (let [excluded-cols #{:dx :dy}
        numeric-cols (filter #(and (not (excluded-cols %))
                                   (every? number? (tc/column ds %)))
                             (tc/column-names ds))
        ds-new (ds/select-columns ds numeric-cols)]
    (tc/dataset (map #(product-augmented-map %) (ds/rows ds-new)))))



(defn train-loop
  [ds]
  (let [ds-dx (tc/drop-missing (tc/drop-columns ds [:dy]))
        ds-dy (tc/drop-missing (tc/drop-columns ds [:dx]))
        split-x (first (tc/split->seq ds-dx :holdout {:seed 112723 :ratio 0.8}))
        split-y (first (tc/split->seq ds-dy :holdout {:seed 112723 :ratio 0.8}))
        pipeline-x (mm/pipeline
                    (ds-mm/set-inference-target :dx)
                    #:metamorph{:id :model}
                    (ml/model {:model-type :smile.regression/lasso, :lambda (double 0.05) :max-iterations (int 500000)})) 
        pipeline-y (mm/pipeline
                    (ds-mm/set-inference-target :dy)
                    #:metamorph{:id :model}
                    (ml/model {:model-type :smile.regression/lasso, :lambda (double 0.05) :max-iterations (int 500000)}))
        fitted-x (mm/fit (:train split-x) pipeline-x)
        fitted-y (mm/fit (:train split-y) pipeline-y)]
    [fitted-x fitted-y pipeline-x pipeline-y]))

(defn predict-loop
  [ds model-dx model-dy pipeline-x pipeline-y]
  (let [ds-dx-drop (tc/drop-columns ds [:dy])
        ds-dy-drop (tc/drop-columns ds [:dx])
        ds-dx (tc/drop-missing ds-dx-drop)
        ds-dy (tc/drop-missing ds-dy-drop)
        split-x (first (tc/split->seq ds-dx :holdout {:seed 112723 :ratio 0.8}))
        split-y (first (tc/split->seq ds-dy :holdout {:seed 112723 :ratio 0.8}))
        prediction-x (-> (:test split-x) (mm/transform-pipe pipeline-x model-dx) :metamorph/data :dx)
        prediction-y (-> (:test split-y) (mm/transform-pipe pipeline-y model-dy) :metamorph/data :dy)]
    [prediction-x prediction-y]))

(defn mean-absolute-error
  [y-true y-pred]
  (stats/mean (map #(Math/abs (- %1 %2)) y-true y-pred)))

(defn eval-model
  [ds]
  (let [[fitted-x fitted-y pipeline-x pipeline-y] (train-loop ds)
        [pred-x pred-y] (predict-loop ds fitted-x fitted-y pipeline-x pipeline-y)
        true-x (tc/column (:test (first (tc/split->seq (tc/drop-missing (tc/drop-columns ds [:dy])) :holdout {:seed 112723}))) :dx)
        true-y (tc/column (:test (first (tc/split->seq (tc/drop-missing (tc/drop-columns ds [:dx])) :holdout {:seed 112723}))) :dy)
        mae-x (mean-absolute-error true-x pred-x)
        mae-y (mean-absolute-error true-y pred-y)
        _ (print "model : lasso" "\nmean absolute error (dx) : " mae-x "\nmean absolute error (dy) : " mae-y "\n---------------------------------\n")]
    {:model "lasso" :mae-x mae-x :mae-y mae-y :model-dx fitted-x :model-dy fitted-y :pipeline-dx pipeline-x :pipeline-dy pipeline-y}))

(comment

  (eval-model ds-intermediate)

  (def ds-spl (add-mult-combos-to-dataset ds-intermediate))

  (defn add-columns [df1 df2]
    (reduce
     (fn [acc col]
       (tc/add-column acc col (df2 col)))
     df1
     (tc/column-names df2)))

  (def ds-result (add-columns ds-intermediate ds-spl))
  (def ds-final (tc/add-column ds-result :ones (repeat (tc/row-count ds-result) 1)))
  (tc/shape ds-final)

  (eval-model ds-result)

  (def df
    (tc/dataset [{:a 1 :b 2 :c 3 :dx 1 :dy 2}
                 {:a 4 :b 5 :c 6 :dx 3 :dy 4}]))

  (def df-a
    (add-mult-combos-to-dataset df))

  (def df-b (add-columns df-a df))
  (print df-b)





  )

>