//! Prints the parsed product/assembly tree of a STEP file.
//!
//! ```text
//! cargo run -p step-assemblies --example inspect -- "tests/Sample AP203 Data Files/as1_pe.stp"
//! ```

use step_assemblies::{Node, parse_assembly_file};

fn print_node(node: &Node, depth: usize) {
    let translation = node.world_transform.translation();
    println!(
        "{:indent$}{} [{}] local=({:.4}, {:.4}, {:.4}) world=({:.4}, {:.4}, {:.4})",
        "",
        node.name,
        node.product_name,
        node.local_transform.translation()[0],
        node.local_transform.translation()[1],
        node.local_transform.translation()[2],
        translation[0],
        translation[1],
        translation[2],
        indent = depth * 2,
    );
    for child in &node.children {
        print_node(child, depth + 1);
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: inspect <file.step>");
    match parse_assembly_file(&path) {
        Ok(assembly) => {
            println!(
                "products={} definitions={} representations={} product_definition_shapes={} \
                 roots={} nodes={}",
                assembly.products.len(),
                assembly.definitions.len(),
                assembly.representations.len(),
                assembly.product_definition_shapes.len(),
                assembly.roots.len(),
                assembly.node_count(),
            );
            for warning in &assembly.warnings {
                println!("warning: {warning}");
            }
            for root in &assembly.roots {
                print_node(root, 0);
            }
        }
        Err(error) => {
            eprintln!("{path}: {error}");
            std::process::exit(1);
        }
    }
}
