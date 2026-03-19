SquaredHingeLoss <- function(pred_tensor, label_tensor, margin=1){
  if(length(margin) != 1 || margin < 0){
    stop("margin must be a non-negative scalar")
  }
  yhat = pred_tensor$flatten()
  y = label_tensor$flatten()
  is_positive = y == 1
  is_negative = !is_positive
  v = yhat + is_negative$to(dtype=yhat$dtype) * margin
  s = torch::torch_argsort(v)
  yhat_s = yhat[s]
  is_positive_s = is_positive[s]
  I_pos = torch::torch_where(is_positive_s, 1, 0)$to(dtype=yhat_s$dtype)
  I_neg = 1 - I_pos
  z = (margin - yhat_s) * I_pos
  a = I_pos$cumsum(dim=1)
  b = (2 * z)$cumsum(dim=1)
  c = (z^2)$cumsum(dim=1)
  L_i = (a * yhat_s^2 + b * yhat_s + c) * I_neg
  torch::torch_sum(L_i)
}
