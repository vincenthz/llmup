use gpui::{
    App, Context, Entity, InteractiveElement as _, IntoElement, Menu, MenuItem, MouseButton,
    ParentElement as _, Render, SharedString, Styled as _, Window, div,
};
use gpui_component::{
    IconName, Sizable as _, ThemeMode, ThemeRegistry, TitleBar,
    button::{Button, ButtonVariants as _},
    menu::AppMenuBar,
};

use crate::{
    About, CloseWindow, OpenWebsite, Quit, SelectLocale,
    themes::{SwitchTheme, SwitchThemeMode},
};

pub struct AppTitleBar {
    app_menu_bar: Entity<gpui_component::menu::AppMenuBar>,
}

impl AppTitleBar {
    pub fn new(
        title: impl Into<SharedString>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        app_menus_init(title, cx);

        let app_menu_bar = AppMenuBar::new(window, cx);

        Self { app_menu_bar }
    }
}

pub fn app_menus_init(title: impl Into<SharedString>, cx: &mut App) {
    cx.set_menus(<[_]>::into_vec(Box::new([
        (Menu {
            name: title.into(),
            items: vec![
                MenuItem::action("About", About),
                MenuItem::Separator,
                MenuItem::action("Open...", OpenWebsite),
                MenuItem::Separator,
                MenuItem::Submenu(Menu {
                    name: "Appearance".into(),
                    items: vec![
                        MenuItem::action("Light", SwitchThemeMode(ThemeMode::Light)),
                        MenuItem::action("Dark", SwitchThemeMode(ThemeMode::Dark)),
                    ],
                }),
                theme_menu(cx),
                language_menu(cx),
                MenuItem::Separator,
                MenuItem::action("Quit", Quit),
            ],
        }),
        (Menu {
            name: "Edit".into(),
            items: vec![
                MenuItem::action("Undo", gpui_component::input::Undo),
                MenuItem::action("Redo", gpui_component::input::Redo),
                MenuItem::separator(),
                MenuItem::action("Cut", gpui_component::input::Cut),
                MenuItem::action("Copy", gpui_component::input::Copy),
                MenuItem::action("Paste", gpui_component::input::Paste),
                MenuItem::separator(),
                MenuItem::action("Delete", gpui_component::input::Delete),
                MenuItem::action(
                    "Delete Previous Word",
                    gpui_component::input::DeleteToPreviousWordStart,
                ),
                MenuItem::action(
                    "Delete Next Word",
                    gpui_component::input::DeleteToNextWordEnd,
                ),
                MenuItem::separator(),
                MenuItem::action("Find", gpui_component::input::Search),
                MenuItem::separator(),
                MenuItem::action("Select All", gpui_component::input::SelectAll),
            ],
        }),
        (Menu {
            name: "Window".into(),
            items: vec![
                MenuItem::action("Close Window", CloseWindow),
                //MenuItem::separator(),
                //MenuItem::action("Toggle Search", ToggleSearch),
            ],
        }),
        (Menu {
            name: "Help".into(),
            items: vec![MenuItem::action("Open Website", OpenWebsite)],
        }),
    ])));
}

impl Render for AppTitleBar {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        TitleBar::new()
            .child(div().flex().items_center().child(self.app_menu_bar.clone()))
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_end()
                    .px_2()
                    .gap_2()
                    .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
                    // add the github button of the project
                    .child(
                        Button::new("github")
                            .icon(IconName::GitHub)
                            .small()
                            .ghost()
                            .on_click(|_, _, cx| cx.open_url("https://github.com/vincenthz/llmup")),
                    ),
            )
    }
}

fn language_menu(_cx: &App) -> MenuItem {
    MenuItem::Submenu(Menu {
        name: "Language".into(),
        items: vec![MenuItem::action("English", SelectLocale("en".into()))],
    })
}

fn theme_menu(cx: &App) -> MenuItem {
    let themes = ThemeRegistry::global(cx).sorted_themes();
    MenuItem::Submenu(Menu {
        name: "Theme".into(),
        items: themes
            .iter()
            .map(|theme| MenuItem::action(theme.name.clone(), SwitchTheme(theme.name.clone())))
            .collect(),
    })
}
