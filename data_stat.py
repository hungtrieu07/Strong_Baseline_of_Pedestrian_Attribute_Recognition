str = "personalLess30,personalLess45,personalLess60,personalLarger60,carryingBackpack,carryingOther,lowerBodyCasual,upperBodyCasual,lowerBodyFormal,upperBodyFormal,accessoryHat,upperBodyJacket,lowerBodyJeans,footwearLeatherShoes,upperBodyLogo,hairLong,personalMale,carryingMessengerBag,accessoryMuffler,accessoryNothing,carryingNothing,upperBodyPlaid,carryingPlasticBags,footwearSandals,footwearShoes,lowerBodyShorts,upperBodyShortSleeve,lowerBodyShortSkirt,footwearSneaker,upperBodyThinStripes,accessorySunglasses,lowerBodyTrousers,upperBodyTshirt,upperBodyOther,upperBodyVNeck,upperBodyBlack,upperBodyBlue,upperBodyBrown,upperBodyGreen,upperBodyGrey,upperBodyOrange,upperBodyPink,upperBodyPurple,upperBodyRed,upperBodyWhite,upperBodyYellow,lowerBodyBlack,lowerBodyBlue,lowerBodyBrown,lowerBodyGreen,lowerBodyGrey,lowerBodyOrange,lowerBodyPink,lowerBodyPurple,lowerBodyRed,lowerBodyWhite,lowerBodyYellow,hairBlack,hairBlue,hairBrown,hairGreen,hairGrey,hairOrange,hairPink,hairPurple,hairRed,hairWhite,hairYellow,footwearBlack,footwearBlue,footwearBrown,footwearGreen,footwearGrey,footwearOrange,footwearPink,footwearPurple,footwearRed,footwearWhite,footwearYellow,accessoryHeadphone,personalLess15,carryingBabyBuggy,hairBald,footwearBoots,lowerBodyCapri,carryingShoppingTro,carryingUmbrella,personalFemale,carryingFolder,accessoryHairBand,lowerBodyHotPants,accessoryKerchief,lowerBodyLongSkirt,upperBodyLongSleeve,lowerBodyPlaid,lowerBodyThinStripes,carryingLuggageCase,upperBodyNoSleeve,hairShort,footwearStocking,upperBodySuit,carryingSuitcase,lowerBodySuits,upperBodySweater,upperBodyThickStripes"

# Split the string by commas
items = str.split(',')

# Add numbering to each item
formatted_data = '\n'.join([f"{i + 1}. {item}" for i, item in enumerate(items)])

print(formatted_data)

"""
1. personalLess30
2. personalLess45
3. personalLess60
4. personalLarger60
5. carryingBackpack
11. accessoryHat
16. hairLong
17. personalMale
18. carryingMessengerBag
26. lowerBodyShorts
27. upperBodyShortSleeve
28. lowerBodyShortSkirt
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
81. personalLess15
88. personalFemale
93. lowerBodyLongSkirt
94. upperBodyLongSleeve
97. carryingLuggageCase
99. hairShort
102. carryingSuitcase
"""