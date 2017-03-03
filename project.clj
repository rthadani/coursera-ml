(defproject coursera-ml "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [net.mikera/core.matrix "0.57.0"]
                 [incanter "1.5.7" :exclusions [net.mikera/core.matrix]]
                 [cheshire "5.6.3"]
                 [org.clojars.ds923y/nd4clj "0.1.1-SNAPSHOT" :exclusions [net.mikera/core.matrix]]
                 [prismatic/plumbing "0.5.3"]]
  :profiles {:dev {:dependencies [[net.mikera/core.matrix "0.53.0" :classifier "tests"]] }}
  )
