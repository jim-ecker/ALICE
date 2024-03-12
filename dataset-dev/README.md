# ALICE Dataset Development 

## Existing Work

* [Extractive Research Slide Generation Using Windowed Labeling Ranking](https://github.com/atharsefid/Extractive_Research_Slide_Generation_Using_Windowed_Labeling_Ranking)
* [Automatic Slide Generation from Scientific Papers](https://www.kaggle.com/datasets/andrewmvd/automatic-slide-generation-from-scientific-papers)
* [Sciduet Dataset](https://github.com/IBM/document2slides)
* [Document Visual Qustion Answering on Multiple Images](https://doi.org/10.48550/arXiv.2301.04883)

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
