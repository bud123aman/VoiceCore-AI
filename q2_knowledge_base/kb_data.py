"""
Synthetic health insurance knowledge base records.
In production: replace with scraped/parsed real documents.
Schema mirrors the assessment example exactly.
"""

KB_RECORDS = [
    {
        "record_id": "kb_product_001",
        "title": "HealthShield Basic Plan Overview",
        "content": (
            "HealthShield Basic covers hospitalization, emergency care, and outpatient consultations. "
            "The plan is available for individuals aged 18–60. Annual premium starts at ₹8,000 for "
            "individuals and ₹14,000 for family floater (up to 4 members). Sum insured options: "
            "₹3L, ₹5L, ₹10L. Pre-existing disease waiting period is 2 years. No co-payment required "
            "for network hospitals."
        ),
        "category": "product_overview",
        "source": "product_brochure_v2024",
        "version": "2.1",
        "pii": False,
    },
    {
        "record_id": "kb_product_002",
        "title": "HealthShield Premium Plan Overview",
        "content": (
            "HealthShield Premium offers comprehensive coverage including OPD, dental, vision, and "
            "maternity benefits. Available for ages 18–65. Annual premium starts at ₹18,000 individual, "
            "₹28,000 family floater (up to 6 members). Sum insured: ₹10L, ₹25L, ₹50L. "
            "Pre-existing disease waiting period: 1 year. Includes international emergency coverage up "
            "to $10,000. Free annual health check-up included. Room rent: single private AC room."
        ),
        "category": "product_overview",
        "source": "product_brochure_v2024",
        "version": "2.1",
        "pii": False,
    },
    {
        "record_id": "kb_product_003",
        "title": "HealthShield Senior Plan",
        "content": (
            "HealthShield Senior is designed for individuals aged 60–80. Covers hospitalization, "
            "day-care procedures, domiciliary treatment, and AYUSH treatment. Premium starts at "
            "₹22,000/year. Sum insured: ₹3L, ₹5L. Pre-existing disease coverage from day 1 for "
            "declared conditions after medical underwriting. Co-payment: 20% for all claims. "
            "No medical test required up to age 65."
        ),
        "category": "product_overview",
        "source": "product_brochure_v2024",
        "version": "2.1",
        "pii": False,
    },

    {
        "record_id": "kb_policy_001",
        "title": "Eligibility Criteria — Age and Health",
        "content": (
            "Basic Plan: 18–60 years. Premium Plan: 18–65 years. Senior Plan: 60–80 years. "
            "Children can be added as dependents from 91 days old. Applicants with pre-existing "
            "conditions such as diabetes, hypertension, or heart disease are eligible but subject to "
            "a 2-year waiting period (Basic) or 1-year waiting period (Premium) before those "
            "conditions are covered. Conditions not disclosed at application are not covered. "
            "BMI above 35 requires medical underwriting."
        ),
        "category": "qualification_rules",
        "source": "policy_document_v2024",
        "version": "3.0",
        "pii": False,
    },
    {
        "record_id": "kb_policy_002",
        "title": "Waiting Periods — What Is and Is Not Covered",
        "content": (
            "30-day initial waiting period: no claims except accidents in first 30 days. "
            "2-year waiting period: pre-existing diseases (Basic Plan). "
            "1-year waiting period: pre-existing diseases (Premium Plan). "
            "9-month waiting period: maternity benefits (Premium Plan only). "
            "2-year waiting period: specific named diseases — cataract, hernia, knee replacement, "
            "joint replacement, varicose veins, sinusitis. "
            "Accidents are covered from day 1 with no waiting period on any plan."
        ),
        "category": "policy_rules",
        "source": "policy_document_v2024",
        "version": "3.0",
        "pii": False,
    },
    {
        "record_id": "kb_policy_003",
        "title": "Claim Process — Cashless and Reimbursement",
        "content": (
            "Cashless claims: only at network hospitals (8,500+ hospitals pan-India). "
            "Pre-authorization required for planned hospitalization — submit at least 72 hours before. "
            "Emergency cashless: notify within 24 hours of admission. "
            "Reimbursement claims: submit within 30 days of discharge with original bills, discharge "
            "summary, prescription, and lab reports. Claims are settled within 7 working days of "
            "complete documentation. Partial claims are possible if some expenses are excluded."
        ),
        "category": "policy_rules",
        "source": "policy_document_v2024",
        "version": "3.0",
        "pii": False,
    },
    {
        "record_id": "kb_policy_004",
        "title": "Exclusions — What Is Not Covered",
        "content": (
            "Permanent exclusions (all plans): cosmetic surgery, self-inflicted injuries, war injuries, "
            "experimental treatments, infertility treatments (Basic), alcohol or drug-related illness, "
            "HIV/AIDS treatment (unless acquired through blood transfusion). "
            "Dental: only covered under Premium plan for accidents. Routine dental excluded. "
            "Vision: only covered under Premium plan. Spectacles/contact lenses excluded. "
            "OPD: only covered under Premium plan. Consultation fees excluded under Basic."
        ),
        "category": "policy_rules",
        "source": "policy_document_v2024",
        "version": "3.0",
        "pii": False,
    },
    {
        "record_id": "kb_policy_005",
        "title": "Premium Calculation Factors",
        "content": (
            "Premium is calculated based on: age of oldest member (family floater), sum insured chosen, "
            "city of residence (Zone A: metro cities — higher premium; Zone B: other cities), "
            "pre-existing conditions (loading of 10–30% may apply), number of members covered, "
            "and plan type. No-claim bonus: 5% sum insured increase per claim-free year, up to 50%. "
            "Premium payment modes: annual (5% discount), semi-annual, quarterly, monthly. "
            "GST of 18% applicable on all premiums."
        ),
        "category": "policy_rules",
        "source": "policy_document_v2024",
        "version": "3.0",
        "pii": False,
    },

    {
        "record_id": "kb_faq_001",
        "title": "FAQ — Can I add family members later?",
        "content": (
            "Yes. You can add spouse and children at renewal time. Adding parents requires a fresh "
            "application under the family floater plan if they are below 60, or the Senior plan "
            "if above 60. Mid-term additions are allowed only for newborns (within 90 days of birth) "
            "and newly married spouses (within 60 days of marriage). A fresh waiting period does NOT "
            "restart for the original policyholder when adding new members."
        ),
        "category": "faq",
        "source": "faq_document_v2024",
        "version": "1.5",
        "pii": False,
    },
    {
        "record_id": "kb_faq_002",
        "title": "FAQ — What happens if I miss a premium payment?",
        "content": (
            "A grace period of 30 days is provided for annual and semi-annual policies, and 15 days "
            "for monthly policies. Claims during the grace period are valid only if premium is paid "
            "before claim settlement. If premium is not paid within the grace period, the policy lapses. "
            "Reinstatement is possible within 90 days of lapse with payment of outstanding premium and "
            "a declaration of good health. Post-90 days, a fresh policy application is required and "
            "waiting periods restart."
        ),
        "category": "faq",
        "source": "faq_document_v2024",
        "version": "1.5",
        "pii": False,
    },
    {
        "record_id": "kb_faq_003",
        "title": "FAQ — Is there a free-look period?",
        "content": (
            "Yes. A 15-day free-look period is available from the date of receipt of the policy "
            "document. During this period, you can return the policy if you disagree with any terms. "
            "Refund will be the full premium minus proportionate risk premium for the period on cover "
            "and stamp duty charges. Free-look period applies only on new policies, not renewals. "
            "To cancel, submit a written request to any branch or email cancellations@healthshield.in."
        ),
        "category": "faq",
        "source": "faq_document_v2024",
        "version": "1.5",
        "pii": False,
    },
    {
        "record_id": "kb_faq_004",
        "title": "FAQ — Does the plan cover COVID-19 and infectious diseases?",
        "content": (
            "Yes. COVID-19 hospitalization is covered under all plans as per IRDAI guidelines. "
            "Home care treatment for COVID is covered under Premium plan if prescribed by a doctor "
            "and treatment requires hospitalization-level care. Infectious diseases like dengue, "
            "malaria, typhoid, and chikungunya are covered from day 31 (after initial waiting period). "
            "Vector-borne diseases require hospitalization of minimum 24 hours for claim eligibility."
        ),
        "category": "faq",
        "source": "faq_document_v2024",
        "version": "1.5",
        "pii": False,
    },
    {
        "record_id": "kb_faq_005",
        "title": "FAQ — Tax benefits on health insurance premium",
        "content": (
            "Premiums paid for health insurance qualify for tax deduction under Section 80D of the "
            "Income Tax Act. Self and family (below 60): deduction up to ₹25,000/year. "
            "Senior citizen (60+): deduction up to ₹50,000/year. If you pay premium for parents "
            "below 60: additional ₹25,000 deduction. Parents above 60: additional ₹50,000 deduction. "
            "Maximum combined deduction possible: ₹1,00,000/year. Deduction applies only on non-cash "
            "payment modes (online, cheque, credit card)."
        ),
        "category": "faq",
        "source": "faq_document_v2024",
        "version": "1.5",
        "pii": False,
    },

    {
        "record_id": "kb_objection_001",
        "title": "Objection — Premium is too expensive",
        "content": (
            "Acknowledge the concern. Reframe cost vs risk: a single hospitalization in a metro city "
            "costs ₹80,000–₹3,00,000 on average. The annual premium is a fraction of that risk. "
            "Offer alternatives: Basic plan at ₹8,000/year (₹667/month) provides core coverage. "
            "Mention tax benefits under Section 80D which effectively reduce the out-of-pocket cost. "
            "Suggest lower sum insured option to reduce premium. Offer to check eligibility for "
            "group/employer top-up if applicable. Do not pressure — offer a callback to discuss."
        ),
        "category": "objection_handling",
        "source": "sales_playbook_v2024",
        "version": "2.0",
        "pii": False,
    },
    {
        "record_id": "kb_objection_002",
        "title": "Objection — I already have employer-provided insurance",
        "content": (
            "Validate the point — employer insurance is a good starting benefit. Then explain the gaps: "
            "employer insurance typically covers only hospitalization and ends when you leave the job. "
            "Personal policy is portable and continuous. Family members may not be covered under "
            "employer plans, or coverage may be low (₹1–2L). Personal plan covers OPD, dental, vision "
            "under Premium. Suggest a top-up or super top-up plan as a cost-effective supplement that "
            "kicks in above the employer coverage threshold. Do not dismiss employer coverage."
        ),
        "category": "objection_handling",
        "source": "sales_playbook_v2024",
        "version": "2.0",
        "pii": False,
    },
    {
        "record_id": "kb_objection_003",
        "title": "Objection — I am young and healthy, I don't need insurance",
        "content": (
            "Acknowledge good health — that's exactly the right time to buy insurance. Explain: "
            "buying young locks in lower premiums for life (premiums increase with age). "
            "Pre-existing conditions that develop later won't be covered if you wait. "
            "Accidents and sudden illness can happen at any age. Share statistic: 30% of health "
            "insurance claims are from people under 35. Waiting periods start from day of purchase — "
            "buying now means full coverage is active sooner. Frame it as financial planning, not fear."
        ),
        "category": "objection_handling",
        "source": "sales_playbook_v2024",
        "version": "2.0",
        "pii": False,
    },
    {
        "record_id": "kb_objection_004",
        "title": "Objection — Claims are always rejected / insurance companies don't pay",
        "content": (
            "Acknowledge the concern — there are bad actors in the industry. Then provide facts: "
            "our claim settlement ratio is 97.3% (FY2023-24, as per IRDAI annual report). "
            "Explain the most common rejection reasons: non-disclosure of pre-existing conditions, "
            "claiming during waiting period, treatment at non-network hospital without pre-auth. "
            "All of these are avoidable with proper understanding of the policy. "
            "Offer to walk through the policy terms clearly. Mention IRDAI grievance redressal if "
            "any dispute arises. Do not make claims that cannot be verified."
        ),
        "category": "objection_handling",
        "source": "sales_playbook_v2024",
        "version": "2.0",
        "pii": False,
    },

    {
        "record_id": "kb_qualify_001",
        "title": "Lead Qualification Criteria — High Priority",
        "content": (
            "Mark lead as HIGH PRIORITY if: age 25–50, employed or self-employed with stated income, "
            "no current health insurance or coverage below ₹5L, has dependents (spouse/children), "
            "expressed interest in specific plan, willing to share health details. "
            "Action: offer immediate quote, schedule follow-up call within 24 hours, offer document "
            "assistance for application. Assign to senior agent for conversion."
        ),
        "category": "qualification_rules",
        "source": "crm_playbook_v2024",
        "version": "1.2",
        "pii": False,
    },
    {
        "record_id": "kb_qualify_002",
        "title": "Lead Qualification Criteria — Medium Priority",
        "content": (
            "Mark lead as MEDIUM PRIORITY if: age 18–24 or 51–60, has some existing coverage but "
            "looking to upgrade, asked about pricing but not committed, has pre-existing conditions "
            "requiring underwriting, or is comparing with competitors. "
            "Action: send brochure and comparison document via WhatsApp/email, follow up in 48 hours, "
            "offer free health consultation call with medical advisor."
        ),
        "category": "qualification_rules",
        "source": "crm_playbook_v2024",
        "version": "1.2",
        "pii": False,
    },
    {
        "record_id": "kb_qualify_003",
        "title": "Lead Qualification Criteria — Low Priority / Refer Elsewhere",
        "content": (
            "Mark lead as LOW PRIORITY or REFER if: age above 65 (refer to Senior plan specialist), "
            "looking for group/corporate plan (refer to B2B team), income below ₹2L/year and no "
            "ability to pay premiums, or explicitly stated no interest after objection handling. "
            "Do not push sale on clearly uninterested leads. Log reason for low priority and set "
            "6-month follow-up reminder. Offer government scheme information (Ayushman Bharat) if "
            "customer cannot afford private insurance."
        ),
        "category": "qualification_rules",
        "source": "crm_playbook_v2024",
        "version": "1.2",
        "pii": False,
    },

    {
        "record_id": "kb_escalation_001",
        "title": "When to Escalate to Human Agent",
        "content": (
            "Escalate immediately to human agent if: customer explicitly asks for human, "
            "customer is angry or distressed, medical emergency is mentioned, customer mentions "
            "an active claim issue or rejection, customer wants to cancel an existing policy, "
            "customer asks legal or regulatory question beyond standard FAQ, or customer provides "
            "conflicting health information requiring underwriting review. "
            "Escalation phrase: 'I'm connecting you with our specialist who can assist you better. "
            "Please hold for a moment.' Never argue with customer before escalating."
        ),
        "category": "escalation_rules",
        "source": "ops_handbook_v2024",
        "version": "1.0",
        "pii": False,
    },

    {
        "record_id": "kb_network_001",
        "title": "Network Hospitals — How to Find",
        "content": (
            "HealthShield has 8,500+ empanelled network hospitals across India. "
            "To find the nearest network hospital: visit healthshield.in/network-hospitals, "
            "use the mobile app (HealthShield App on iOS and Android), or call our helpline "
            "1800-XXX-XXXX (toll-free, 24x7). Search by city, pincode, or hospital name. "
            "Network includes all major government hospitals, Apollo, Fortis, Max, Manipal, "
            "Narayana, and most district hospitals. Cashless facility only at network hospitals."
        ),
        "category": "network_info",
        "source": "website_v2024",
        "version": "1.0",
        "pii": False,
    },

    {
        "record_id": "kb_pii_001",
        "title": "Sample Lead Record — Rajesh Kumar [PII-PROTECTED]",
        "content": (
            "Lead: Rajesh Kumar, DOB: 12/05/1985, Mobile: 98XXXXXXXX, "
            "Email: rajesh.kumar@email.com, Address: 14 MG Road, Bengaluru 560001. "
            "Interested in Premium Plan, family of 4. Pre-existing: Type 2 diabetes. "
            "Assigned agent: Priya S. Follow-up scheduled: 2024-03-15."
        ),
        "category": "lead_record",
        "source": "crm_export_2024",
        "version": "1.0",
        "pii": True,
    },
    {
        "record_id": "kb_pii_002",
        "title": "Sample Claim Record — Anita Sharma [PII-PROTECTED]",
        "content": (
            "Policyholder: Anita Sharma, Policy No: HS-2023-00441, "
            "Aadhaar: XXXX-XXXX-1234 (masked). Claim CLM-8821 filed 2024-01-10. "
            "Diagnosis: Appendectomy. Hospital: Apollo Bengaluru. "
            "Claimed amount: Rs 1,85,000. Settlement: Approved 2024-01-22. "
            "Bank account: HDFC XXXX1234 (masked)."
        ),
        "category": "claim_record",
        "source": "claims_db_export_2024",
        "version": "1.0",
        "pii": True,
    },
]
