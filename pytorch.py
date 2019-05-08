from extract_feature_v2 import extract_feature
from model_irse import IR_50

model = IR_50([112, 112])
extract_feature("nabeel.jpg",model,"backbone_ir50_ms1m_epoch120.pth")