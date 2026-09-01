# lg-aimers-pitch-control
LG Aimers 9기 Phase 2 해커톤: 투구 제구 성공 확률 예측 AI 모델 개발 프로젝트


trackman_history.csv는 못 씀 (중요 발견)

trackman_history.csv로 투수별 상세 특성을 붙여보려 했는데, train.csv의 pitcher_id(약 20700~24633, 792명)와 trackman의 pitcher_trackman_id(약 5만~7176만, 906명)가 전혀 안 겹쳐요. 아마 익명화 방식이 서로 달라서 매칭이 불가능한 것 같아요. 그래서 개인별로는 못 쓰고, 기껏해야 "리그 전체 평균 구속/구종 분포" 같은 아주 거친 참고값 정도만 가능한데, 이득보다 리스크가 커서 일단은 스킵하고 필요하면 나중에 다시 검토하자고 제안했어요.