## University Dataset – Relation Map

### What This Database Represents

This database models a **university course management system**. It is designed to show:
- Which **departments** exist at the university
- Which **instructors** belong to which departments
- Which **students** are enrolled, what they **major** in, and who their **advisor** is
- Which **courses** are offered, who **teaches** them, and in which **classroom** they run
- Which **students take which courses**, in which **semester/year**, and what **grades** they receive

You can think of it as the backend of a simple **student information system (SIS)** or **university portal**.

Typical questions this DB can answer:
- How many students does each department have?
- Which courses does a given instructor teach this semester?
- Which students are enrolled in a specific course?
- What is the grade distribution for a course?
- How full are classrooms (enrollment vs capacity)?
- Which advisors have the most advisees?

### Tables

1. **departments**
   - **PK**: `department_id`
   - Columns: `department_id`, `name`, `building`

2. **instructors**
   - **PK**: `instructor_id`
   - **FK**: `department_id` → `departments.department_id`
   - Columns: `instructor_id`, `first_name`, `last_name`, `email`, `department_id`

3. **students**
   - **PK**: `student_id`
   - **FK1**: `major_department_id` → `departments.department_id`
   - **FK2**: `advisor_instructor_id` → `instructors.instructor_id`
   - Columns: `student_id`, `first_name`, `last_name`, `email`, `major_department_id`, `advisor_instructor_id`, `enrollment_year`

4. **classrooms**
   - **PK**: `classroom_id`
   - Columns: `classroom_id`, `building`, `room_number`, `capacity`

5. **courses**
   - **PK**: `course_id`
   - **FK1**: `department_id` → `departments.department_id`
   - **FK2**: `instructor_id` → `instructors.instructor_id`
   - **FK3**: `classroom_id` → `classrooms.classroom_id`
   - Columns: `course_id`, `code`, `title`, `department_id`, `instructor_id`, `classroom_id`, `credits`

6. **enrollments**
   - **PK**: `enrollment_id`
   - **FK1**: `student_id` → `students.student_id`
   - **FK2**: `course_id` → `courses.course_id`
   - Columns: `enrollment_id`, `student_id`, `course_id`, `semester`, `year`, `grade`

---

### Relationship Overview

#### Departments
- **departments 1 — n instructors**
- **departments 1 — n students** (via `major_department_id`)
- **departments 1 — n courses**

#### Instructors
- **instructors 1 — n courses**
- **instructors 1 — n students** (as advisors)

#### Classrooms
- **classrooms 1 — n courses**

#### Students & Courses
- **students n — n courses** via `enrollments`
- `students.student_id` = `enrollments.student_id`
- `courses.course_id` = `enrollments.course_id`

---

### ASCII ER Diagram (simplified)

```text
        +-----------------+
        |  departments    |
        |-----------------|
        | department_id PK|
        | name            |
        | building        |
        +--------+--------+
                 | 1
     1           |           1
     |           |           |
     v           v           v
+----------+  +--------------+    +----------------+
|instructors|  |   students   |    |    courses     |
|-----------|  |--------------|    |----------------|
|instructor_|  |student_id  PK|    |course_id    PK |
|id PK      |  |major_depart_|    |department_id FK|
|departm_id |  |_ment_id  FK |    |instructor_id FK|
|FK         |  |advisor_inst_|    |classroom_id  FK|
+-----+-----+  |_ructor_id FK|    +--------+-------+
      | 1              | 1                 |
      |                |                   | 1
      |                |                   v
      |                |           +---------------+
      |                |           |  classrooms   |
      |                |           |---------------|
      |                |           |classroom_id PK|
      |                |           |building       |
      |                |           |room_number    |
      |                |           |capacity       |
      |                |           +---------------+
      |                |
      |                | n
      v                v
          +-------------------------+
          |       enrollments       |
          |-------------------------|
          | enrollment_id PK        |
          | student_id    FK        |
          | course_id     FK        |
          | semester                |
          | year                    |
          | grade                   |
          +-------------------------+
```

---

### How to Use This Map

- When creating tables in SQL, define **primary keys** as shown above.
- Add **FOREIGN KEY** constraints matching the arrows in the diagram.
- Use this map to write **JOIN queries**, e.g.:
  - Students → Enrollments → Courses → Instructors → Departments
  - Departments → Courses → Enrollments → Students
