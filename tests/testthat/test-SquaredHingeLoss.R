library(testthat)
data.table::setDTthreads(1L)

naive_squared_hinge_loss <- function(pred, label, margin=1){
  pos <- which(label == 1)
  neg <- which(label != 1)
  if(length(pos) == 0 || length(neg) == 0) return(0)
  total <- 0
  for(j in pos){
    for(k in neg){
      total <- total + max(margin - (pred[j] - pred[k]), 0)^2
    }
  }
  total
}

if(torch::torch_is_installed()){

  test_that("SquaredHingeLoss rejects invalid margin", {
    pred <- torch::torch_tensor(c(1.0, -1.0))
    label <- torch::torch_tensor(c(1, -1))
    expect_error(
      mlr3torchAUM::SquaredHingeLoss(pred, label, margin=-1),
      "non-negative")
  })

  test_that("SquaredHingeLoss matches naive O(n^2) implementation", {
    n <- 10
    pred <- rnorm(n)
    label <- rep(c(1L, -1L), length.out=n)     
    fast_loss <- torch::as_array(
      mlr3torchAUM::SquaredHingeLoss(
        torch::torch_tensor(pred, dtype=torch::torch_double()),
        torch::torch_tensor(label)
      )
    )
    slow_loss <- naive_squared_hinge_loss(pred, label)
    expect_equal(fast_loss, slow_loss)
  })

}
