import org.apache.log4j.Level
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.{GBTClassifier, LinearSVC, LogisticRegression}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.ml.feature.{LabeledPoint, OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.rdd.RDD



object Mushroom {

  def get_metrics(predictions: DataFrame): Unit ={
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("class").setRawPredictionCol("rawPrediction")
    val accuracy = evaluator.evaluate(predictions)

    println(s"Prediction Accuracy = $accuracy")

    val preds = predictions.select("prediction").rdd.map(row => row.getDouble(0))
    val labels = predictions.select("class").rdd.map(row => row.getInt(0).toDouble)
    val predictionAndLabels: RDD[(Double, Double)] = preds.zip(labels)

    val metrics = new BinaryClassificationMetrics(predictionAndLabels)

    // Precision by threshold
    val precision = metrics.precisionByThreshold
    precision.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }

    // Recall by threshold
    val recall = metrics.recallByThreshold
    recall.foreach { case (t, r) =>
      println(s"Threshold: $t, Recall: $r")
    }

    // Precision-Recall Curve
    val PRC = metrics.pr

    // F-measure
    val f1Score = metrics.fMeasureByThreshold
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 1")
    }

    val beta = 0.5
    val fScore = metrics.fMeasureByThreshold(beta)
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 0.5")
    }

    // AUPRC
    val auPRC = metrics.areaUnderPR
    println("Area under precision-recall curve = " + auPRC)

    // Compute thresholds used in ROC and PR curves
    val thresholds = precision.map(_._1)

    // ROC Curve
    val roc = metrics.roc

    // AUROC
    val auROC = metrics.areaUnderROC
    println("Area under ROC = " + auROC)
  }

  def predict(spark:SparkSession, training: DataFrame, evaluation:DataFrame, indexers: StringIndexer, encoders: OneHotEncoder, assembler: VectorAssembler, algorithm: PipelineStage,file_name: String): Unit ={

    val pipeline = new Pipeline().setStages(Array(indexers, encoders, assembler, algorithm))
    val model_fit = pipeline.fit(training)
    val predictions = model_fit.transform(evaluation)

    println("Training Metrics")
    get_metrics(predictions)

    // extract the test data
    val test_data = spark.read.option("header", true).option("inferSchema", true).csv("data/mushroom_test.csv")

    // exclude column that contains a question mark
    val test_df = test_data.drop("stalk-root")

    val test_predictions = model_fit.transform(test_df) //make predictions test data

    test_predictions.groupBy("prediction").count().show()

    val results = test_predictions
      .select("id", "prediction")
      .withColumn("prediction", test_predictions.col("prediction")
        .cast(IntegerType)).withColumnRenamed("prediction", "class")


    results.coalesce(1).write.format("com.databricks.spark.csv")
      .option("header", "true")
      .mode("overwrite")
      .save(s"data/submission/${file_name}")
  }
  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder
      .appName("MushroomML")
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "file:///C:/temp") // Necessary to work around a Windows bug in Spark 2.0.0; omit if you're not on Windows.
      .getOrCreate()

    spark.sparkContext.setLogLevel(Level.ERROR.toString())

    // Load data
    val data = spark.read.option("header", true).option("inferSchema", true).csv("data/mushroom_train.csv")

    //inspect schema
    data.printSchema()

    //inspect data
    data.show()

    //inspect rows that contain null values
    data.filter(row => row.anyNull).show

    data.describe().show()

    println(s"number of duplicate IDs: ${data.groupBy("id").count.filter("count > 1").count()}")

    //display the unique values in each column
    for (column <- data.columns) {
      val unqiue_values = data.select(column).distinct()
      println(s"Column $column:")
      unqiue_values.show()
    }

    //exclude column that contains a question mark
    val cleaned_data = data.drop("stalk-root")
//    val cleaned_data = data.filter(!col("stalk-root").contains('?'))

    // split data into training and eval
    val Array(training, evaluation) = cleaned_data.randomSplit(Array(0.8, 0.2), seed = 10)

    val indexers = new StringIndexer()
      .setInputCols(training.columns.filter(!_.matches("class|id")))
      .setOutputCols(training.columns.filter( !_.matches("class|id")).map(name => s"${name}_index"))

    val encoders = new OneHotEncoder()
      .setInputCols(indexers.getOutputCols.filter(_ != "veil-type_index"))
      .setOutputCols(indexers.getOutputCols.filter(_ != "veil-type_index").map( name => s"${name}_vector"))

    val assembler = new VectorAssembler()
      .setInputCols(encoders.getOutputCols).setOutputCol("features")


    val logistic_regression_model = new LogisticRegression()
      .setFeaturesCol("features").setLabelCol("class")

    val gradient_boosted_tree = new GBTClassifier()
      .setLabelCol("class")
      .setFeaturesCol("features")

    val linear_support_vector_machine = new LinearSVC().setLabelCol("class").setFeaturesCol("features")

    println("\nLogistic Regression")
    predict(spark,training,evaluation,indexers,encoders,assembler,logistic_regression_model,"logistic-regression.csv")
    println("\nGradient Boosted Tree")
    predict(spark,training,evaluation,indexers,encoders,assembler,gradient_boosted_tree,"gradient-boosted-tree.csv")
    println("\nLinear Support Vector Machine")
    predict(spark, training, evaluation, indexers, encoders, assembler, linear_support_vector_machine, "linear-support-vector-machine.csv")


  }

}
