# RL-Scheduling
Integrade RL, Optimization on schduling problem

### Experiment Seting
1. 50 Delievery Nodes (Factories)
2. 1 Truck Workshop
3. Logistics vehicles number: 12
4. Product list:

list of raw matrerial
| Factory id | Component | Ratio(t/s) |
|------------|---------|------------|
| 0 | P0 | 0.01 |
| 1 | P1 | 0.01 |
| ... | ... | ... |
| 44 | P44 | 0.01 |

| Factory id | Product | Ratio(t/s) | Value ($/t) | Raw Material|
|------------|---------|------------|-------------|-------------|
| 45 | A | 0.01 | 10| P0,P1,P2,P3,P4,P5,P6,P7,P8 |
| 46 | B | 0.02 | 7.5 | P9,P10,P11,P12,P13,P14,P15,P16,P17 |
| 47 | C | 0.05 | 5 | P18,P19,P20,P21,P22,P23,P24,P25,P26 |
| 48 | D | 0.03 | 2.5 | P27,P28,P29,P30,P31,P32,P33,P34,P35 |
| 49 | E | 0.04 | 3 | P36,P37,P38,P39,P40,P41,P42,P43,P44 |


### Matainence & Repair Seting
1. Matainence time: 6 hours
2. Repair time (broken): 2 days
3. Repair action: remove from the map, after it is recover, it will start from a random factory.