(defproject cyclone-track-forecasting "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [org.clojars.haifengl/smile "4.2.0"]
                 [org.scicloj/scicloj.ml.xgboost "6.3.0"]
                 [org.scicloj/metamorph.ml "1.2"]
                 [org.scicloj/clay "2-beta19"]
                 [scicloj/tablecloth "7.042"]
                 [org.scicloj/kindly "4-beta15"]]
  :main ^:skip-aot cyclone-track-forecasting.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all
                       :jvm-opts ["-Dclojure.compiler.direct-linking=true"]}})
