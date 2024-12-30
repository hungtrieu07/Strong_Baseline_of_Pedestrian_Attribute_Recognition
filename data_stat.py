str = "personalLess30,personalLess45,personalLess60,personalLarger60,carryingBackpack,carryingOther,lowerBodyCasual,upperBodyCasual,lowerBodyFormal,upperBodyFormal,accessoryHat,upperBodyJacket,lowerBodyJeans,footwearLeatherShoes,upperBodyLogo,hairLong,personalMale,carryingMessengerBag,accessoryMuffler,accessoryNothing,carryingNothing,upperBodyPlaid,carryingPlasticBags,footwearSandals,footwearShoes,lowerBodyShorts,upperBodyShortSleeve,lowerBodyShortSkirt,footwearSneaker,upperBodyThinStripes,accessorySunglasses,lowerBodyTrousers,upperBodyTshirt,upperBodyOther,upperBodyVNeck,upperBodyBlack,upperBodyBlue,upperBodyBrown,upperBodyGreen,upperBodyGrey,upperBodyOrange,upperBodyPink,upperBodyPurple,upperBodyRed,upperBodyWhite,upperBodyYellow,lowerBodyBlack,lowerBodyBlue,lowerBodyBrown,lowerBodyGreen,lowerBodyGrey,lowerBodyOrange,lowerBodyPink,lowerBodyPurple,lowerBodyRed,lowerBodyWhite,lowerBodyYellow,hairBlack,hairBlue,hairBrown,hairGreen,hairGrey,hairOrange,hairPink,hairPurple,hairRed,hairWhite,hairYellow,footwearBlack,footwearBlue,footwearBrown,footwearGreen,footwearGrey,footwearOrange,footwearPink,footwearPurple,footwearRed,footwearWhite,footwearYellow,accessoryHeadphone,personalLess15,carryingBabyBuggy,hairBald,footwearBoots,lowerBodyCapri,carryingShoppingTro,carryingUmbrella,personalFemale,carryingFolder,accessoryHairBand,lowerBodyHotPants,accessoryKerchief,lowerBodyLongSkirt,upperBodyLongSleeve,lowerBodyPlaid,lowerBodyThinStripes,carryingLuggageCase,upperBodyNoSleeve,hairShort,footwearStocking,upperBodySuit,carryingSuitcase,lowerBodySuits,upperBodySweater,upperBodyThickStripes"

# Split the string by commas
items = str.split(',')

# Add numbering to each item
formatted_data = '\n'.join([f"{i + 1}. {item}" for i, item in enumerate(items)])

print(formatted_data)

"""
p1. personalLess30
2. personalLess45
3. personalLess60
4. personalLarger60
5. carryingBackpack
6. carryingOther
7. lowerBodyCasual
8. upperBodyCasual
9. lowerBodyFormal
10. upperBodyFormal
11. accessoryHat
12. upperBodyJacket
13. lowerBodyJeans
14. footwearLeatherShoes
15. upperBodyLogo
16. hairLong
17. personalMale
18. carryingMessengerBag
19. accessoryMuffler
20. accessoryNothing
21. carryingNothing
22. upperBodyPlaid
23. carryingPlasticBags
24. footwearSandals
25. footwearShoes
26. lowerBodyShorts
27. upperBodyShortSleeve
28. lowerBodyShortSkirt
29. footwearSneaker
30. upperBodyThinStripes
31. accessorySunglasses
32. lowerBodyTrousers
33. upperBodyTshirt
34. upperBodyOther
35. upperBodyVNeck
36. upperBodyBlack
37. upperBodyBlue
38. upperBodyBrown
39. upperBodyGreen
40. upperBodyGrey
41. upperBodyOrange
42. upperBodyPink
43. upperBodyPurple
44. upperBodyRed
45. upperBodyWhite
46. upperBodyYellow
47. lowerBodyBlack
48. lowerBodyBlue
49. lowerBodyBrown
50. lowerBodyGreen
51. lowerBodyGrey
52. lowerBodyOrange
53. lowerBodyPink
54. lowerBodyPurple
55. lowerBodyRed
56. lowerBodyWhite
57. lowerBodyYellow
58. hairBlack
59. hairBlue
60. hairBrown
61. hairGreen
62. hairGrey
63. hairOrange
64. hairPink
65. hairPurple
66. hairRed
67. hairWhite
68. hairYellow
69. footwearBlack
70. footwearBlue
71. footwearBrown
72. footwearGreen
73. footwearGrey
74. footwearOrange
75. footwearPink
76. footwearPurple
77. footwearRed
78. footwearWhite
79. footwearYellow
80. accessoryHeadphone
81. personalLess15
82. carryingBabyBuggy
83. hairBald
84. footwearBoots
85. lowerBodyCapri
86. carryingShoppingTro
87. carryingUmbrella
88. personalFemale
89. carryingFolder
90. accessoryHairBand
91. lowerBodyHotPants
92. accessoryKerchief
93. lowerBodyLongSkirt
94. upperBodyLongSleeve
95. lowerBodyPlaid
96. lowerBodyThinStripes
97. carryingLuggageCase
98. upperBodyNoSleeve
99. hairShort
100. footwearStocking
101. upperBodySuit
102. carryingSuitcase
103. lowerBodySuits
104. upperBodySweater
105. upperBodyThickStripes
"""