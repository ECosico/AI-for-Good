def test_load():
  return 'loaded'
  
def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def cond_prob(table, evidence, evidence_value, target, target_value,):
  t_subset = up_table_subset(table, target, 'equals', target_value)
  e_list = up_get_column(t_subset, evidence)
  p_b_a = sum([1 if v==evidence_value else 0 for v in e_list])/len(e_list)
  return p_b_a

def cond_probs_product(table, evidence_row, target, target_value):
  evidence_columns_table = up_drop_column(table, target)
  evidence_columns  = up_list_column_names(evidence_columns_table)
  return up_product([cond_prob(table, e[0],e[1], target, target_value) for e in up_zip_lists(evidence_columns, evidence_row)])

def prior_prob(table, target, target_value):
  t_list = up_get_column(table, target)
  p_a = sum([1 if v==target_value else 0 for v in t_list])/len(t_list)
  return p_a

for row in last10_rows:
  actual = row[-1]
  row = row[:-1]
  probabilities = naive_bayes(shelter_table, row, target)
  prediction = 1 if probabilities[1]>=.5 else 0
  print(f'Actual: {actual}, Prediction: {prediction},  Probabilities: {probabilities}')
