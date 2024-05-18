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

def naive_bayes(table, evidence_row, target):
  #compute P(target=0|...) by using cond_probs_product, take the product of the list, finally multiply by P(target=0) using prob_cond on that
  neg = cond_probs_product(table, evidence_row, target, 0) * prior_prob(table, target, 0)
  #do same for P(target=1|...)
  pos = cond_probs_product(table, evidence_row, target, 1)* prior_prob(table, target, 1)

  #Use compute_probs to get 2 probabilities
  compute_probs(neg,pos)
  #return your 2 results in a list
  return compute_probs(neg,pos)
