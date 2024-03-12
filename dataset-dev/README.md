# ALICE Dataset Development 

## Existing Work

* [Extractive Research Slide Generation Using Windowed Labeling Ranking](https://github.com/atharsefid/Extractive_Research_Slide_Generation_Using_Windowed_Labeling_Ranking)
* [Automatic Slide Generation from Scientific Papers](https://www.kaggle.com/datasets/andrewmvd/automatic-slide-generation-from-scientific-papers)
* [Sciduet Dataset](https://github.com/IBM/document2slides)
* [Document Visual Qustion Answering on Multiple Images](https://doi.org/10.48550/arXiv.2301.04883)

## Example of Presentation with related document listed

```json
{
      "_meta": {
        "score": 75.3385
      },
      "copyright": {
        "determinationType": "PUBLIC_USE_PERMITTED",
        "thirdPartyContentCondition": "NOT_SET"
      },
      "subjectCategories": [
        "Spacecraft Design, Testing And Performance",
        "Communications And Radar",
        "Lasers And Masers"
      ],
      "exportControl": {
        "isExportControl": "NO",
        "ear": "NO",
        "itar": "NO"
      },
      "distributionDate": "2019-07-11T00:00:00.0000000+00:00",
      "otherReportNumbers": [
        "NF1676L-19084"
      ],
      "fundingNumbers": [
        {
          "number": "WBS 743588.01.99.99.99.99.23",
          "type": "WBS"
        }
      ],
      "title": "Lidar Sensor Performance in Closed-Loop Flight Testing of the Morpheus Rocket-Propelled Lander to a Lunar-Like Hazard Field",
      "stiType": "CONFERENCE_PAPER",
      "distribution": "PUBLIC",
      "submittedDate": "2016-05-06T10:59:49.8800000+00:00",
      "authorAffiliations": [
        {
          "sequence": 0,
          "submissionId": 20160005928,
          "meta": {
            "author": {
              "name": "Roback, Vincent E."
            },
            "organization": {
              "name": "NASA Langley Research Center",
              "location": "Hampton, VA, United States"
            }
          },
          "id": "dbbc22325ae6476ca33756ea17d9ea27"
        },
        {
          "sequence": 1,
          "submissionId": 20160005928,
          "meta": {
            "author": {
              "name": "Pierrottet, Diego F."
            },
            "organization": {
              "name": "NASA Langley Research Center",
              "location": "Hampton, VA, United States"
            }
          },
          "id": "c7a4f473c3c143db813313e9cccf60a3"
        },
        {
          "sequence": 2,
          "submissionId": 20160005928,
          "meta": {
            "author": {
              "name": "Amzajerdian, Farzin"
            },
            "organization": {
              "name": "NASA Langley Research Center",
              "location": "Hampton, VA, United States"
            }
          },
          "id": "aa3ab30bfa1c4cc3a75689d938df09bf"
        },
        {
          "sequence": 3,
          "submissionId": 20160005928,
          "meta": {
            "author": {
              "name": "Barnes, Bruce W."
            },
            "organization": {
              "name": "NASA Langley Research Center",
              "location": "Hampton, VA, United States"
            }
          },
          "id": "44b00bb7822c4bdd8571c2c5ece09198"
        },
        {
          "sequence": 4,
          "submissionId": 20160005928,
          "meta": {
            "author": {
              "name": "Hines, Glenn D."
            },
            "organization": {
              "name": "NASA Langley Research Center",
              "location": "Hampton, VA, United States"
            }
          },
          "id": "56f997560a334f22a6d58ec8af6c42f6"
        },
        {
          "sequence": 5,
          "submissionId": 20160005928,
          "meta": {
            "author": {
              "name": "Petway, Larry B."
            },
            "organization": {
              "name": "NASA Langley Research Center",
              "location": "Hampton, VA, United States"
            }
          },
          "id": "20ab3ef528644185a77196838bc68f03"
        },
        {
          "sequence": 6,
          "submissionId": 20160005928,
          "meta": {
            "author": {
              "name": "Brewster, Paul F."
            },
            "organization": {
              "name": "NASA Langley Research Center",
              "location": "Hampton, VA, United States"
            }
          },
          "id": "b7ee78e8512845469eda3adf1c8e8e69"
        },
        {
          "sequence": 7,
          "submissionId": 20160005928,
          "meta": {
            "author": {
              "name": "Kempton, Kevin S."
            },
            "organization": {
              "name": "NASA Langley Research Center",
              "location": "Hampton, VA, United States"
            }
          },
          "id": "b7b59741dd604bc5a3e2db8379487466"
        },
        {
          "sequence": 8,
          "submissionId": 20160005928,
          "meta": {
            "author": {
              "name": "Bulyshev, Alexander E."
            },
            "organization": {
              "name": "Analytical Mechanics Associates, Inc.",
              "location": "Hampton, VA, United States"
            }
          },
          "id": "f79d89e875f842e996ce4c64e3912304"
        }
      ],
      "stiTypeDetails": "Conference Paper",
      "technicalReviewType": "TECHNICAL_REVIEW_TYPE_NONE",
      "modified": "2022-11-19T16:39:27.8290800+00:00",
      "id": 20160005928,
      "legacyMeta": {
        "__type": "LegacyMetaIndex, StrivesApi.ServiceModel",
        "accessionNumber": ""
      },
      "created": "2016-05-06T10:59:49.8800000+00:00",
      "center": {
        "code": "LaRC",
        "name": "Langley Research Center",
        "id": "1e229fe5b7284965a153b0f761643383"
      },
      "onlyAbstract": true,
      "sensitiveInformation": 2,
      "abstract": "For the first time, a suite of three lidar sensors have been used in flight to scan a lunar-like hazard field, identify a safe landing site, and, in concert with an experimental Guidance, Navigation, and Control (GN&C) system, guide the Morpheus autonomous, rocket-propelled, free-flying test bed to a safe landing on the hazard field. The lidar sensors and GN&C system are part of the Autonomous Precision Landing and Hazard Detection and Avoidance Technology (ALHAT) project which has been seeking to develop a system capable of enabling safe, precise crewed or robotic landings in challenging terrain on planetary bodies under any ambient lighting conditions. The 3-D imaging flash lidar is a second generation, compact, real-time, air-cooled instrument developed from a number of cutting-edge components from industry and NASA and is used as part of the ALHAT Hazard Detection System (HDS) to scan the hazard field and build a 3-D Digital Elevation Map (DEM) in near-real time for identifying safe sites. The flash lidar is capable of identifying a 30 cm hazard from a slant range of 1 km with its 8 cm range precision at 1 sigma. The flash lidar is also used in Hazard Relative Navigation (HRN) to provide position updates down to a 250m slant range to the ALHAT navigation filter as it guides Morpheus to the safe site. The Doppler Lidar system has been developed within NASA to provide velocity measurements with an accuracy of 0.2 cm/sec and range measurements with an accuracy of 17 cm both from a maximum range of 2,200 m to a minimum range of several meters above the ground. The Doppler Lidar's measurements are fed into the ALHAT navigation filter to provide lander guidance to the safe site. The Laser Altimeter, also developed within NASA, provides range measurements with an accuracy of 5 cm from a maximum operational range of 30 km down to 1 m and, being a separate sensor from the flash lidar, can provide range along a separate vector. The Laser Altimeter measurements are also fed into the ALHAT navigation filter to provide lander guidance to the safe site. The flight tests served as the culmination of the TRL 6 journey for the lidar suite and included launch from a pad situated at the NASA-Kennedy Space Center Shuttle Landing Facility (SLF) runway, a lunar-like descent trajectory from an altitude of 250m, and landing on a lunar-like hazard field of rocks, craters, hazardous slopes, and safe sites 400m down-range just off the North end of the runway. The tests both confirmed the expected performance and also revealed several challenges present in the flight-like environment which will feed into future TRL advancement of the sensors. The flash lidar identified hazards as small as 30 cm from the maximum slant range of 450 m which Morpheus could provide, however, it was occasionally susceptible to an increase in range noise due to heated air from the Morpheus rocket plume which entered its Field-of-View (FOV). The flash lidar was also susceptible to pre-triggering on dust during the HRN phase which was created during launch and transported by the wind. The Doppler Lidar provided velocity and range measurements to the expected accuracy levels yet it was also susceptible to signal degradation due to air heated by the rocket engine. The Laser Altimeter, operating with a degraded transmitter laser, also showed signal attenuation over a few seconds at a specific phase of the flight due to the heat plume generated by the rocket engine.",
      "isLessonsLearned": false,
      "disseminated": "DOCUMENT_AND_METADATA",
      "meetings": [
        {
          "country": "United States",
          "submissionId": 20160005928,
          "endDate": "2015-01-09T00:00:00.0000000+00:00",
          "sponsors": [
            {
              "meta": {
                "organization": {
                  "name": "American Inst. of Aeronautics and Astronautics",
                  "location": "Reston, VA, United States"
                }
              },
              "meetingId": "5d3165d0297243a5866db5c2fd0a9533",
              "id": "1238983a80764dd5acfba2d8c356cf4c"
            }
          ],
          "name": "SciTech 2015",
          "location": "Kissimmee, FL",
          "id": "5d3165d0297243a5866db5c2fd0a9533",
          "startDate": "2015-01-05T00:00:00.0000000+00:00"
        }
      ],
      "publications": [
        {
          "submissionId": 20160005928,
          "id": "b5cb8cc3dc844e20a64647710ea415b8",
          "publicationDate": "2015-01-05T00:00:00.0000000+00:00"
        }
      ],
      "status": "CURATED",
      "related": [
        {
          "disseminated": "DOCUMENT_AND_METADATA",
          "id": 20160006855,
          "type": "SEE_ALSO",
          "title": "Lidar Sensor Performance in Closed-Loop Flight Testing of the Morpheus Rocket-Propelled Lander to a Lunar-Like Hazard Field",
          "stiType": "PRESENTATION",
          "distribution": "PUBLIC",
          "status": "CURATED"
        }
      ],
      "downloads": [
        {
          "draft": false,
          "mimetype": "application/pdf",
          "name": "20160005928.pdf",
          "type": "STI",
          "links": {
            "original": "/api/citations/20160005928/downloads/20160005928.pdf",
            "pdf": "/api/citations/20160005928/downloads/20160005928.pdf",
            "fulltext": "/api/citations/20160005928/downloads/20160005928.txt"
          }
        }
      ],
      "downloadsAvailable": true,
      "index": "submissions-2024-03-07-05-31"
    }
```

## Problem Aligning Presentations to Publications (My presentation/publication pair is retrieved using two different author names)

### Example API Response author=James%20Ecker

```json
{
  "stats": {
    "took": 36,
    "total": 1,
    "estimate": false,
    "maxScore": 0
  },
  "results": [
    {
      "_meta": {
        "score": 0
      },
      "copyright": {
        "disclosedInvention": false,
        "thirdPartyPermissionsProduced": true,
        "licenseType": "NO",
        "containsThirdPartyMaterial": true,
        "thirdPartyLocationComments": "Slides 15, 16, 17, 19, 20, 21",
        "containsIndication": false,
        "determinationType": "MAY_INCLUDE_COPYRIGHT_MATERIAL",
        "thirdPartyContentCondition": "NOT_SET"
      },
      "subjectCategories": [
        "Cybernetics, Artificial Intelligence And Robotics"
      ],
      "exportControl": {
        "isExportControl": "NO",
        "ear": "NO",
        "itar": "NO"
      },
      "distributionDate": "2021-02-04T05:00:00.0000000+00:00",
      "otherReportNumbers": [],
      "fundingNumbers": [
        {
          "number": "533127.02.60.07",
          "type": "WBS"
        }
      ],
      "title": "Synthetic Data Generation for 3D Mesh Prediction and Spatial Reasoning During Multi-Agent Robotic Missions",
      "stiType": "PRESENTATION",
      "distribution": "PUBLIC",
      "submittedDate": "2020-12-10T12:12:54.8734650+00:00",
      "authorAffiliations": [
        {
          "organizationId": "68603dff261e5d2d9de5a550aa9e6b30",
          "sequence": 0,
          "submissionId": 20205011392,
          "meta": {
            "author": {
              "orcidId": "",
              "name": "James Ecker"
            },
            "organization": {
              "name": "Langley Research Center",
              "location": "Hampton, Virginia, United States"
            }
          },
          "id": "8a46b613c63842c1b24dfd36319d7a5e",
          "userType": "CIVIL",
          "userId": "388e33a0d1b4427fb2dbe2fd22644124"
        },
        {
          "organizationId": "68603dff261e5d2d9de5a550aa9e6b30",
          "sequence": 1,
          "submissionId": 20205011392,
          "meta": {
            "author": {
              "orcidId": "",
              "name": "Benjamin Kelley"
            },
            "organization": {
              "name": "Langley Research Center",
              "location": "Hampton, Virginia, United States"
            }
          },
          "id": "f1b4053a11e64deeabb7127b2ac1965e",
          "userType": "CIVIL",
          "userId": "dd0b9c806a5a469db81b96d9b1846608"
        },
        {
          "organizationId": "68603dff261e5d2d9de5a550aa9e6b30",
          "sequence": 2,
          "submissionId": 20205011392,
          "meta": {
            "author": {
              "orcidId": "",
              "name": "Danette Allen"
            },
            "organization": {
              "name": "Langley Research Center",
              "location": "Hampton, Virginia, United States"
            }
          },
          "id": "c6b4c29c3d344f878a9b50b4b8a2d40c",
          "userType": "CIVIL",
          "userId": "1dd9ecdcb213459c9cdfb8a88b25dbdd"
        }
      ],
      "stiTypeDetails": "Presentation",
      "technicalReviewType": "NASA_TECHNICAL_MANAGEMENT",
      "modified": "2023-07-06T23:33:07.8827120+00:00",
      "id": 20205011392,
      "sourceIdentifiers": [],
      "created": "2020-12-10T12:00:29.0621670+00:00",
      "center": {
        "code": "LaRC",
        "name": "Langley Research Center",
        "id": "1e229fe5b7284965a153b0f761643383"
      },
      "onlyAbstract": false,
      "sensitiveInformation": 2,
      "abstract": "In-space assembly operations require accurate reasoning over the pose, location, and structural organization of both the autonomous agents and assembly materials. In a full six-degree-of-freedom space, an accurate understanding of the full three-dimensional structure of the object of interest greatly enriches information for pose estimation and collision planning. Current methods of predicting pose estimation require a priori understanding of the shape of the object. Additionally, visual information in the space environment is impacted by variations in contrast and illumination. Using synthetic data allows us to rapidly generate large datasets with in varying environments and lighting conditions. This work details the generation of synthetic data used to explore the use of a region-based convolutional neural networks to detect objects of interest and predict a voxel-based three-dimensional mesh in order to understand their full three-dimensional shape. This mesh provides useful spatial information during in-space assembly operations without requiring either the complexity of maintaining models over the progress of building an object or observations from multiple angles. The generated meshes are then compared to that of ground truth in order to measure its performance.",
      "isLessonsLearned": false,
      "disseminated": "DOCUMENT_AND_METADATA",
      "meetings": [
        {
          "country": "US",
          "submissionId": 20205011392,
          "endDate": "2021-01-21T05:00:00.0000000+00:00",
          "sponsors": [
            {
              "organizationId": "71fb4af96d4a59b5a0a2084db7b4ce35",
              "meta": {
                "organization": {
                  "name": "American Institute of Aeronautics and Astronautics",
                  "location": "Reston, Virginia, United States"
                }
              },
              "meetingId": "ca0cfca07def4ebd9ffce1edb03c9baa",
              "id": "394351a9587c4f94b7a878f94fabde39"
            }
          ],
          "name": "AIAA Scitech 2021",
          "location": "Virtual",
          "id": "ca0cfca07def4ebd9ffce1edb03c9baa",
          "url": "https://www.aiaa.org/SciTech?SSO=Y",
          "startDate": "2021-01-11T05:00:00.0000000+00:00"
        }
      ],
      "status": "CURATED",
      "related": [],
      "downloads": [
        {
          "draft": false,
          "mimetype": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
          "name": "Scitech-Presentation-Jim-Ecker v1.1.pptx",
          "type": "STI",
          "links": {
            "original": "/api/citations/20205011392/downloads/Scitech-Presentation-Jim-Ecker%20v1.1.pptx",
            "pdf": "/api/citations/20205011392/downloads/Scitech-Presentation-Jim-Ecker%20v1.1.pptx.pdf",
            "fulltext": "/api/citations/20205011392/downloads/Scitech-Presentation-Jim-Ecker%20v1.1.pptx.txt"
          }
        }
      ],
      "downloadsAvailable": true,
      "index": "submissions-2024-03-07-05-31"
    }
  ],
  "aggregations": {
    "created": {
      "buckets": [
        {
          "key_as_string": "2020",
          "key": 1577836800000,
          "doc_count": 1
        }
      ]
    },
    "author": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": [
        {
          "key": "Benjamin Kelley",
          "doc_count": 1
        },
        {
          "key": "Danette Allen",
          "doc_count": 1
        },
        {
          "key": "James Ecker",
          "doc_count": 1
        }
      ]
    },
    "reportNumber": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": []
    },
    "center": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": [
        {
          "key": "LaRC",
          "doc_count": 1
        }
      ]
    },
    "index": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": [
        {
          "key": "submissions-2024-03-07-05-31",
          "doc_count": 1
        }
      ]
    },
    "fundingNumber": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": [
        {
          "key": "533127.02.60.07",
          "doc_count": 1
        }
      ]
    },
    "published": {
      "buckets": []
    },
    "distribution": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": [
        {
          "key": "PUBLIC",
          "doc_count": 1
        }
      ]
    },
    "stiType": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": [
        {
          "key": "PRESENTATION",
          "doc_count": 1
        }
      ]
    },
    "subjectCategory": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": [
        {
          "key": "Cybernetics, Artificial Intelligence And Robotics",
          "doc_count": 1
        }
      ]
    },
    "disseminated": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": [
        {
          "key": "DOCUMENT_AND_METADATA",
          "doc_count": 1
        }
      ]
    },
    "stiTypeDetails": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": [
        {
          "key": "Presentation",
          "doc_count": 1
        }
      ]
    },
    "organization": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": [
        {
          "key": "Langley Research Center",
          "doc_count": 1
        }
      ]
    },
    "modified": {
      "buckets": [
        {
          "key_as_string": "2023",
          "key": 1672531200000,
          "doc_count": 1
        }
      ]
    },
    "keyword": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": []
    }
  }
```
### Example API Response author=James%20E%20Ecker

```json
{
  "stats": {
    "took": 42,
    "total": 2,
    "estimate": false,
    "maxScore": 0
  },
  "results": [
    {
      "_meta": {
        "score": 0
      },
      "copyright": {
        "disclosedInvention": false,
        "thirdPartyPermissionsProduced": true,
        "licenseType": "NO",
        "containsThirdPartyMaterial": true,
        "containsIndication": false,
        "determinationType": "MAY_INCLUDE_COPYRIGHT_MATERIAL",
        "thirdPartyContentCondition": "NOT_SET"
      },
      "subjectCategories": [
        "Cybernetics, Artificial Intelligence And Robotics"
      ],
      "keywords": [
        "synthetic data",
        "neural network",
        "convolutional network."
      ],
      "exportControl": {
        "isExportControl": "NO",
        "ear": "NO",
        "itar": "NO"
      },
      "distributionDate": "2021-02-04T05:00:00.0000000+00:00",
      "fundingNumbers": [
        {
          "number": " 533127.02.60.07",
          "type": "WBS"
        }
      ],
      "title": "Synthetic Data Generation for 3D Mesh Prediction and Spatial Reasoning During Multi-Agent Robotic Missions",
      "stiType": "CONFERENCE_PAPER",
      "distribution": "PUBLIC",
      "submittedDate": "2020-12-04T15:05:54.7213760+00:00",
      "authorAffiliations": [
        {
          "organizationId": "68603dff261e5d2d9de5a550aa9e6b30",
          "sequence": 0,
          "submissionId": 20205011070,
          "meta": {
            "author": {
              "orcidId": "",
              "name": "James E Ecker"
            },
            "organization": {
              "name": "Langley Research Center",
              "location": "Hampton, Virginia, United States"
            }
          },
          "id": "329e08d63b2646d0a8c850d5a8f256ed",
          "userType": "CIVIL",
          "userId": "388e33a0d1b4427fb2dbe2fd22644124"
        },
        {
          "organizationId": "68603dff261e5d2d9de5a550aa9e6b30",
          "sequence": 1,
          "submissionId": 20205011070,
          "meta": {
            "author": {
              "orcidId": "",
              "name": "Benjamin N Kelley"
            },
            "organization": {
              "name": "Langley Research Center",
              "location": "Hampton, Virginia, United States"
            }
          },
          "id": "2c777ded81034807ab9ee857e910383f",
          "userType": "CIVIL",
          "userId": "dd0b9c806a5a469db81b96d9b1846608"
        },
        {
          "organizationId": "68603dff261e5d2d9de5a550aa9e6b30",
          "sequence": 2,
          "submissionId": 20205011070,
          "meta": {
            "author": {
              "orcidId": "",
              "name": "B Danette Allen"
            },
            "organization": {
              "name": "Langley Research Center",
              "location": "Hampton, Virginia, United States"
            }
          },
          "id": "6e39521c6fdc4fc68c2c504b1f047d6e",
          "userId": "a0a4983807ad45f891a8db5b734c9548"
        }
      ],
      "stiTypeDetails": "Conference Paper",
      "technicalReviewType": "NASA_TECHNICAL_MANAGEMENT",
      "modified": "2023-07-06T23:33:07.8827120+00:00",
      "id": 20205011070,
      "sourceIdentifiers": [],
      "created": "2020-12-04T14:54:55.5010940+00:00",
      "center": {
        "code": "LaRC",
        "name": "Langley Research Center",
        "id": "1e229fe5b7284965a153b0f761643383"
      },
      "onlyAbstract": false,
      "sensitiveInformation": 2,
      "abstract": "In-space assembly operations require accurate reasoning over the pose, location, and structural organization of both the autonomous agents and assembly materials. In a full six-degree-of-freedom space, an accurate understanding of the full three-dimensional structure of the object of interest greatly enriches information for pose estimation and collision planning. Current methods of predicting pose estimation require a priori understanding of the shape of the object. Additionally, visual information in the space environment is impacted by variations in contrast and illumination. Using synthetic data allows us to rapidly generate large datasets with in varying environments and lighting conditions.This work details the generation of synthetic data used to explore the use of a region-based convolutional neural networks to detect objects of interest and predict a voxel-based three-dimensional mesh in order to understand their full three-dimensional shape. This mesh provides useful spatial information during in-space assembly operations without requiring either the complexity of maintaining models over the progress of building an object or observations from multiple angles. The generated meshes are then compared to that of ground truth in order to measure its performance.",
      "isLessonsLearned": false,
      "disseminated": "DOCUMENT_AND_METADATA",
      "meetings": [
        {
          "country": "US",
          "submissionId": 20205011070,
          "endDate": "2021-01-21T05:00:00.0000000+00:00",
          "sponsors": [
            {
              "organizationId": "71fb4af96d4a59b5a0a2084db7b4ce35",
              "meta": {
                "organization": {
                  "name": "American Institute of Aeronautics and Astronautics",
                  "location": "Reston, Virginia, United States"
                }
              },
              "meetingId": "37bb58223f3e4086b4e55e92c07ef311",
              "id": "f9af91fc49294315af85607d45f05baa"
            }
          ],
          "name": "AIAA Scitech 2021",
          "location": "Virtual",
          "id": "37bb58223f3e4086b4e55e92c07ef311",
          "url": "",
          "startDate": "2021-01-11T05:00:00.0000000+00:00"
        }
      ],
      "status": "CURATED",
      "related": [],
      "downloads": [
        {
          "draft": false,
          "mimetype": "application/pdf",
          "name": "Scitech2021_proofv9_Jim_Ecker.pdf",
          "type": "STI",
          "links": {
            "original": "/api/citations/20205011070/downloads/Scitech2021_proofv9_Jim_Ecker.pdf",
            "pdf": "/api/citations/20205011070/downloads/Scitech2021_proofv9_Jim_Ecker.pdf",
            "fulltext": "/api/citations/20205011070/downloads/Scitech2021_proofv9_Jim_Ecker.pdf.txt"
          }
        }
      ],
      "downloadsAvailable": true,
      "index": "submissions-2024-03-07-05-31"
    },
    {
      "_meta": {
        "score": 0
      },
      "copyright": {
        "disclosedInvention": false,
        "licenseType": "NO",
        "containsThirdPartyMaterial": false,
        "containsIndication": false,
        "determinationType": "GOV_PUBLIC_USE_PERMITTED",
        "thirdPartyContentCondition": "NOT_SET"
      },
      "subjectCategories": [
        "Cybernetics, Artificial Intelligence And Robotics"
      ],
      "exportControl": {
        "isExportControl": "NO",
        "ear": "NO",
        "itar": "NO"
      },
      "distributionDate": "2020-05-27T09:59:17.7400000+00:00",
      "otherReportNumbers": [
        "NF1676L-35089"
      ],
      "fundingNumbers": [
        {
          "number": "533127.02.18.07.02",
          "type": "WBS"
        }
      ],
      "title": "Analyzing Natural Language Context in Human-Machine Teaming using Supervised Machine Learning",
      "stiType": "CONFERENCE_PAPER",
      "distribution": "PUBLIC",
      "submittedDate": "2020-04-29T09:39:28.4770000+00:00",
      "authorAffiliations": [
        {
          "organizationId": "68603dff261e5d2d9de5a550aa9e6b30",
          "sequence": 0,
          "submissionId": 20200003097,
          "meta": {
            "author": {
              "orcidId": "",
              "name": "Bryan A Barrows"
            },
            "organization": {
              "name": "Langley Research Center",
              "location": "Hampton, Virginia, United States"
            }
          },
          "id": "bd931122b4cf418f9ea17966ae00676c",
          "userType": "CIVIL",
          "userId": "f9c561b2beae4945b50588e5f5e61971"
        },
        {
          "organizationId": "68603dff261e5d2d9de5a550aa9e6b30",
          "sequence": 1,
          "submissionId": 20200003097,
          "meta": {
            "author": {
              "orcidId": "",
              "name": "Lisa R Le Vie"
            },
            "organization": {
              "name": "Langley Research Center",
              "location": "Hampton, Virginia, United States"
            }
          },
          "id": "1285fa490752402b8fa831dbffc6220d",
          "userType": "CIVIL",
          "userId": "328e9ff013c04d27877909025826418b"
        },
        {
          "organizationId": "68603dff261e5d2d9de5a550aa9e6b30",
          "sequence": 2,
          "submissionId": 20200003097,
          "meta": {
            "author": {
              "orcidId": "",
              "name": "James E Ecker"
            },
            "organization": {
              "name": "Langley Research Center",
              "location": "Hampton, Virginia, United States"
            }
          },
          "id": "bafe0133cb304c66a74ab4ad27ef88b5",
          "userType": "CIVIL",
          "userId": "388e33a0d1b4427fb2dbe2fd22644124"
        },
        {
          "organizationId": "68603dff261e5d2d9de5a550aa9e6b30",
          "sequence": 3,
          "submissionId": 20200003097,
          "meta": {
            "author": {
              "orcidId": "",
              "name": "B Danette Allen"
            },
            "organization": {
              "name": "Langley Research Center",
              "location": "Hampton, Virginia, United States"
            }
          },
          "id": "26f70dd24f08453998b06ad21e92ce9e",
          "userType": "CIVIL",
          "userId": "1dd9ecdcb213459c9cdfb8a88b25dbdd"
        }
      ],
      "stiTypeDetails": "Conference Paper",
      "technicalReviewType": "TECHNICAL_REVIEW_TYPE_NONE",
      "modified": "2022-11-19T16:44:28.0565860+00:00",
      "id": 20200003097,
      "sourceIdentifiers": [],
      "legacyMeta": {
        "__type": "LegacyMetaIndex, StrivesApi.ServiceModel",
        "accessionNumber": ""
      },
      "created": "2020-04-29T09:39:28.4770000+00:00",
      "center": {
        "code": "LaRC",
        "name": "Langley Research Center",
        "id": "1e229fe5b7284965a153b0f761643383"
      },
      "onlyAbstract": false,
      "sensitiveInformation": 2,
      "abstract": "Building a foundation for trustworthiness and trust verification in multi-asset teaming is the research challenge of Autonomy Teaming and TRAjectories for Complex Trusted Operational   Reliability (ATTRACTOR). The Design Reference   Mission (DRM) for ATTRACTOR is a search and rescue mission objective governed by a multi-member team consisting of human and machine operators. A crucial component to the effort is the communication between humans and autonomous agents throughout both planning and execution stages of the mission. Intuitive communication methods and modalities are posited as critical enablers for certifying trust and trustworthiness. This paper reports on the data collection and analysis conducted in support of the Human Informed Natural-language GANs Evaluation (HINGE)project to attain explainable and trusted communication between human-machine assets. Two identically curated image description datasets were acquired for HINGE, both consisting of two unique input modalities (typed vs. verbal) and retrieved in two distinct contexts (general vs.  specific).  The gathered datasets were assessed and compared using Parts-of-Speech (POS)features, sentence similarity metrics, and linguistic analysis. Then, the datasets were modeled and tested separately and in combination with one another using machine learning algorithms.  The comparison and testing results reveal a superior dataset, by which a preferred context and input is understood, for generating image representations of missing persons using a Generative Adversarial Network (GAN).",
      "isLessonsLearned": false,
      "disseminated": "DOCUMENT_AND_METADATA",
      "meetings": [
        {
          "country": "US",
          "submissionId": 20200003097,
          "endDate": "2020-01-10T05:00:00.0000000+00:00",
          "sponsors": [
            {
              "organizationId": "71fb4af96d4a59b5a0a2084db7b4ce35",
              "meta": {
                "organization": {
                  "name": "American Institute of Aeronautics and Astronautics",
                  "location": "Reston, Virginia, United States"
                }
              },
              "meetingId": "e38caff4e6ff4391b8f7f28f114ab2a9",
              "id": "63107f92100141728557711a126e03c5"
            }
          ],
          "name": "AIAA SciTech 2020",
          "location": "Orlando, FL",
          "id": "e38caff4e6ff4391b8f7f28f114ab2a9",
          "startDate": "2020-01-06T05:00:00.0000000+00:00"
        }
      ],
      "status": "CURATED",
      "related": [],
      "downloads": [
        {
          "draft": false,
          "mimetype": "application/pdf",
          "name": "20200003097.pdf",
          "type": "STI",
          "links": {
            "original": "/api/citations/20200003097/downloads/20200003097.pdf",
            "pdf": "/api/citations/20200003097/downloads/20200003097.pdf",
            "fulltext": "/api/citations/20200003097/downloads/20200003097.txt"
          }
        }
      ],
      "downloadsAvailable": true,
      "index": "submissions-2024-03-07-05-31"
    }
  ],
  "aggregations": {
    "created": {
      "buckets": [
        {
          "key_as_string": "2020",
          "key": 1577836800000,
          "doc_count": 2
        }
      ]
    },
    "author": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": [
        {
          "key": "B Danette Allen",
          "doc_count": 2
        },
        {
          "key": "James E Ecker",
          "doc_count": 2
        },
        {
          "key": "Benjamin N Kelley",
          "doc_count": 1
        },
        {
          "key": "Bryan A Barrows",
          "doc_count": 1
        },
        {
          "key": "Lisa R Le Vie",
          "doc_count": 1
        }
      ]
    },
    "reportNumber": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": [
        {
          "key": "NF1676L-35089",
          "doc_count": 1
        }
      ]
    },
    "center": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": [
        {
          "key": "LaRC",
          "doc_count": 2
        }
      ]
    },
    "index": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": [
        {
          "key": "submissions-2024-03-07-05-31",
          "doc_count": 2
        }
      ]
    },
    "fundingNumber": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": [
        {
          "key": " 533127.02.60.07",
          "doc_count": 1
        },
        {
          "key": "533127.02.18.07.02",
          "doc_count": 1
        }
      ]
    },
    "published": {
      "buckets": []
    },
    "distribution": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": [
        {
          "key": "PUBLIC",
          "doc_count": 2
        }
      ]
    },
    "stiType": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": [
        {
          "key": "CONFERENCE_PAPER",
          "doc_count": 2
        }
      ]
    },
    "subjectCategory": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": [
        {
          "key": "Cybernetics, Artificial Intelligence And Robotics",
          "doc_count": 2
        }
      ]
    },
    "disseminated": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": [
        {
          "key": "DOCUMENT_AND_METADATA",
          "doc_count": 2
        }
      ]
    },
    "stiTypeDetails": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": [
        {
          "key": "Conference Paper",
          "doc_count": 2
        }
      ]
    },
    "organization": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": [
        {
          "key": "Langley Research Center",
          "doc_count": 2
        }
      ]
    },
    "modified": {
      "buckets": [
        {
          "key_as_string": "2022",
          "key": 1640995200000,
          "doc_count": 1
        },
        {
          "key_as_string": "2023",
          "key": 1672531200000,
          "doc_count": 1
        }
      ]
    },
    "keyword": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": [
        {
          "key": "convolutional network.",
          "doc_count": 1
        },
        {
          "key": "neural network",
          "doc_count": 1
        },
        {
          "key": "synthetic data",
          "doc_count": 1
        }
      ]
    }
  }
}
```
